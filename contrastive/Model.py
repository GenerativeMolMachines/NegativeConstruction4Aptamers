import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')



class MicroContrastiveModel(nn.Module):
    def __init__(self, input_dim_apt=768, input_dim_mol=768, latent_dim=768, projection_dim=768, dropout=0.4):
        super().__init__()

        # Энкодер для АПТАМЕРОВ (768 → 768, без сжатия)
        self.apt_encoder = nn.Sequential(
            nn.Linear(input_dim_apt, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Энкодер для МОЛЕКУЛ (768 → 768)
        self.mol_encoder = nn.Sequential(
            nn.Linear(input_dim_mol, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.projection = nn.Identity()

    def encode_aptamer(self, x):
        z = self.apt_encoder(x)
        z = self.projection(z)
        return F.normalize(z, dim=-1)

    def encode_molecule(self, x):
        z = self.mol_encoder(x)
        z = self.projection(z)
        return F.normalize(z, dim=-1)

    def forward(self, anchor_smi, positive_apts, negative_apts):
        z_anchor = self.encode_molecule(anchor_smi)
        z_positive = self.encode_aptamer(positive_apts)

        B, K, D = negative_apts.shape
        negatives_flat = negative_apts.view(B * K, D)
        z_neg = self.encode_aptamer(negatives_flat)
        z_neg = z_neg.view(B, K, -1)

        return {
            'z_anchor': z_anchor,
            'z_positive': z_positive,
            'z_negatives': z_neg
        }

    def forward_symmetric(
        self,
        positive_molecules,
        positive_apts,
        negative_apts,
        negative_molecules,
    ):
        """Encode both counterfactual directions for pair-specific supervision."""
        z_mol = self.encode_molecule(positive_molecules)
        z_apt = self.encode_aptamer(positive_apts)

        batch_size, n_apt_neg, apt_dim = negative_apts.shape
        z_apt_neg = self.encode_aptamer(
            negative_apts.reshape(batch_size * n_apt_neg, apt_dim)
        ).reshape(batch_size, n_apt_neg, -1)

        _, n_mol_neg, mol_dim = negative_molecules.shape
        z_mol_neg = self.encode_molecule(
            negative_molecules.reshape(batch_size * n_mol_neg, mol_dim)
        ).reshape(batch_size, n_mol_neg, -1)

        return {
            'z_molecule': z_mol,
            'z_aptamer': z_apt,
            'z_negative_aptamers': z_apt_neg,
            'z_negative_molecules': z_mol_neg,
        }