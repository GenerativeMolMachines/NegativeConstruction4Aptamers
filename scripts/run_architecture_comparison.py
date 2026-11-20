"""Compare MLP, two-tower MLP and CNN on the released ERE pairs."""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aptabench_features import build_aptamer_features, build_molecule_features, feature_spec_for_split
from data import DEFAULT_DATA, load_aptabench_indexed_pairs
from ere_lgbm import _labeled_eval, make_folds

METHODS = ("composition_shuffle", "random_pairing", "learned_generated", "experimental")
ARCHITECTURES = ("mlp", "two_tower_mlp", "cnn_dual_encoder")


def one_hot(sequences, width=128):
    alphabet = dict(zip("ACGT", range(4)))
    result = np.zeros((len(sequences), 4, width), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for column, base in enumerate(sequence[:width]):
            if base in alphabet:
                result[row, alphabet[base], column] = 1
    return result


def tower(width):
    return nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU())


class Classifier(nn.Module):
    def __init__(self, name, aptamer_width, molecule_width):
        super().__init__()
        self.name = name
        if name == "mlp":
            self.head = nn.Sequential(tower(aptamer_width + molecule_width), nn.Linear(64, 1))
        elif name == "two_tower_mlp":
            self.aptamer = tower(aptamer_width)
        else:
            self.aptamer = nn.Sequential(nn.Conv1d(4, 64, 5, padding=2), nn.ReLU(),
                                         nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
                                         nn.AdaptiveMaxPool1d(1), nn.Flatten())
        if name != "mlp":
            self.molecule = tower(molecule_width)
            self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, aptamer, molecule):
        if self.name == "mlp":
            values = torch.cat((aptamer, molecule), dim=1)
        else:
            a, m = self.aptamer(aptamer), self.molecule(molecule)
            values = torch.cat((a, m, a * m, torch.abs(a - m)), dim=1)
        return self.head(values).squeeze(-1)


def train_predict(name, inputs, labels, n_train, epochs, seed, device):
    torch.manual_seed(seed)
    aptamer, molecule = (torch.as_tensor(x, dtype=torch.float32) for x in inputs)
    train = TensorDataset(aptamer[:n_train], molecule[:n_train], torch.from_numpy(labels[:n_train]))
    batches = DataLoader(train, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = Classifier(name, aptamer.shape[1], molecule.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for a, m, y in batches:
            optimizer.zero_grad()
            loss_fn(model(a.to(device), m.to(device)), y.to(device)).backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return torch.cat([
            model(a.to(device), m.to(device)).sigmoid().cpu()
            for a, m in DataLoader(TensorDataset(aptamer[n_train:], molecule[n_train:]), batch_size=256)
        ]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/architecture_comparison_v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    pools = ROOT / "data/constructed_negatives/ere_fold_local_v2/by_fold"
    data, _ = load_aptabench_indexed_pairs(DEFAULT_DATA)
    seed = 42
    for split in ("random", "molecule"):
        feature_spec = feature_spec_for_split(split)
        for fold, bundle in enumerate(make_folds(data, split, 5, seed)):
            test = _labeled_eval(bundle.test_pos, bundle.test_neg_exp)
            for method in METHODS:
                pool = pools / split / f"fold{fold}" / "train" / f"{method}.csv"
                checksum = hashlib.sha256(pool.read_bytes()).hexdigest()
                train = _labeled_eval(bundle.train_pos, pd.read_csv(pool))
                frame = pd.concat([train, test], ignore_index=True)
                aptamer = build_aptamer_features(frame.sequence.tolist(), feature_spec.aptamer_encoder)
                molecule = build_molecule_features(frame.canonical_smiles.tolist(), data,
                                                   feature_spec.molecule_encoder, 2048, 2)
                n_train = len(train)
                scale = lambda x: StandardScaler().fit(x[:n_train]).transform(x)
                features = scale(aptamer), scale(molecule)
                labels = frame.label.to_numpy(dtype=np.float32)
                for architecture in ARCHITECTURES:
                    output = args.out / f"{split}_{fold}_{method}_{architecture}_e{args.epochs}.csv"
                    if output.exists():
                        continue
                    inputs = (one_hot(frame.sequence.tolist()), features[1]) if architecture == "cnn_dual_encoder" else features
                    scores = train_predict(architecture, inputs, labels, n_train, args.epochs, seed, args.device)
                    row = dict(split_type=split, fold=fold, method=method, architecture=architecture,
                               epochs=args.epochs, seed=seed, n_train=n_train, n_test=len(test),
                               pool_sha256=checksum, roc_auc=roc_auc_score(test.label, scores),
                               ap=average_precision_score(test.label, scores))
                    pending = output.with_suffix(".tmp")
                    pd.DataFrame([row]).to_csv(pending, index=False)
                    pending.replace(output)
                    print(row, flush=True)


if __name__ == "__main__":
    main()
