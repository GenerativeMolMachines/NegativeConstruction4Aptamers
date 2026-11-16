import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class TemperatureScaledLoss(nn.Module):
    def __init__(
        self,
        init_temperature=0.25,
        margin=0.15,
        margin_weight=0.5,
        # Kept for experiments; defaults OFF after RSAPred regression with abs-hinge.
        abs_neg_threshold=0.0,
        abs_neg_weight=0.0,
        abs_neg_mode='hard',
        pos_floor=0.70,
        pos_floor_weight=0.0,
        pair_bce_weight=0.5,
        pair_rank_weight=0.25,
    ):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.margin = margin
        self.margin_weight = margin_weight
        self.abs_neg_threshold = float(abs_neg_threshold)
        self.abs_neg_weight = float(abs_neg_weight)
        self.abs_neg_mode = str(abs_neg_mode)
        self.pos_floor = float(pos_floor)
        self.pos_floor_weight = float(pos_floor_weight)
        self.pair_bce_weight = float(pair_bce_weight)
        self.pair_rank_weight = float(pair_rank_weight)

    def get_temperature(self):
        return torch.exp(self.log_temperature).clamp(0.07, 0.5)

    def compute_margin_loss(self, z_anchor, z_positive, z_negatives):
        """Push positives above hardest negative by a cosine margin (768d space)."""
        pos_sim = torch.sum(z_anchor * z_positive, dim=1)
        if z_negatives is None:
            return torch.zeros((), device=z_anchor.device)

        neg_sim = torch.einsum('bd,bkd->bk', z_anchor, z_negatives).max(dim=1).values
        return F.relu(neg_sim - pos_sim + self.margin).mean()

    def compute_pos_floor_loss(self, z_anchor, z_positive):
        if self.pos_floor_weight <= 0:
            return torch.zeros((), device=z_anchor.device)
        pos_cos = torch.sum(z_anchor * z_positive, dim=1)
        return F.relu(self.pos_floor - pos_cos).mean()

    def compute_pair_bce_loss(self, z_anchor, z_positive, z_negatives, temp):
        """Pair-level BCE on cosine scores — aligns training with fixed-pair AUC."""
        if self.pair_bce_weight <= 0:
            return torch.zeros((), device=z_anchor.device)

        pos_cos = torch.sum(z_anchor * z_positive, dim=1)
        logits = [pos_cos / temp]
        targets = [torch.ones_like(pos_cos)]
        if z_negatives is not None:
            neg_cos = torch.einsum('bd,bkd->bk', z_anchor, z_negatives)
            logits.append(neg_cos.reshape(-1) / temp)
            targets.append(torch.zeros(neg_cos.numel(), device=z_anchor.device))
        return F.binary_cross_entropy_with_logits(
            torch.cat(logits), torch.cat(targets)
        )

    def compute_pair_rank_loss(self, z_anchor, z_positive, z_negatives):
        """Pairwise ranking: every negative should sit below its positive by margin."""
        if self.pair_rank_weight <= 0 or z_negatives is None:
            return torch.zeros((), device=z_anchor.device)

        pos_cos = torch.sum(z_anchor * z_positive, dim=1, keepdim=True)
        neg_cos = torch.einsum('bd,bkd->bk', z_anchor, z_negatives)
        return F.relu(neg_cos - pos_cos + self.margin).mean()

    def compute_abs_neg_loss(self, z_anchor, z_negatives):
        if z_negatives is None or self.abs_neg_weight <= 0:
            zero = torch.zeros((), device=z_anchor.device)
            hard = float('nan')
            if z_negatives is not None:
                neg_cos = torch.einsum('bd,bkd->bk', z_anchor, z_negatives)
                hard = float(neg_cos.max(dim=1).values.mean().item())
                frac = float((neg_cos > self.abs_neg_threshold).float().mean().item())
            else:
                frac = 0.0
            return zero, {
                'abs_neg_loss': 0.0,
                'hard_neg_mean': hard,
                'frac_neg_above_thresh': frac,
            }

        neg_cos = torch.einsum('bd,bkd->bk', z_anchor, z_negatives)
        hard = neg_cos.max(dim=1).values
        if self.abs_neg_mode == 'all':
            abs_loss = F.relu(neg_cos - self.abs_neg_threshold).mean()
        else:
            abs_loss = F.relu(hard - self.abs_neg_threshold).mean()

        frac_above = (neg_cos > self.abs_neg_threshold).float().mean()
        return abs_loss, {
            'abs_neg_loss': float(abs_loss.item()),
            'hard_neg_mean': float(hard.mean().item()),
            'frac_neg_above_thresh': float(frac_above.item()),
        }

    def forward(self, z_anchor, z_positive, z_negatives):
        B = z_anchor.size(0)
        temp = self.get_temperature()

        pos_sim = torch.sum(z_anchor * z_positive, dim=1, keepdim=True) / temp

        if z_negatives is not None:
            neg_sim = torch.einsum('bd,bkd->bk', z_anchor, z_negatives) / temp
            logits = torch.cat([pos_sim, neg_sim], dim=1)
        else:
            logits = pos_sim

        labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
        infonce_loss = F.cross_entropy(logits, labels)
        margin_loss = self.compute_margin_loss(z_anchor, z_positive, z_negatives)
        abs_neg_loss, abs_metrics = self.compute_abs_neg_loss(z_anchor, z_negatives)
        pos_floor_loss = self.compute_pos_floor_loss(z_anchor, z_positive)
        pair_bce_loss = self.compute_pair_bce_loss(
            z_anchor, z_positive, z_negatives, temp
        )
        pair_rank_loss = self.compute_pair_rank_loss(
            z_anchor, z_positive, z_negatives
        )
        loss = (
            infonce_loss
            + self.margin_weight * margin_loss
            + self.abs_neg_weight * abs_neg_loss
            + self.pos_floor_weight * pos_floor_loss
            + self.pair_bce_weight * pair_bce_loss
            + self.pair_rank_weight * pair_rank_loss
        )

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            top3 = (torch.topk(logits, min(3, logits.size(1)), dim=1)
                    .indices == 0).any(dim=1).float().mean()
            pos_cos_mean = torch.sum(z_anchor * z_positive, dim=1).mean()

        return loss, {
            'accuracy': acc.item(),
            'top3_acc': top3.item(),
            'temperature': temp.item(),
            'infonce_loss': infonce_loss.item(),
            'margin_loss': margin_loss.item(),
            'pos_floor_loss': float(pos_floor_loss.item()),
            'pair_bce_loss': float(pair_bce_loss.item()),
            'pair_rank_loss': float(pair_rank_loss.item()),
            'pos_cos_mean': float(pos_cos_mean.item()),
            **abs_metrics,
        }
