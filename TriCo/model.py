from typing import Dict, Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config


def encoder(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(config.DROPOUT),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.GELU(),
    )


def decoder(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(config.DROPOUT),
        nn.Linear(hidden_dim, output_dim),
    )


class TriCo(nn.Module):
    def __init__(
        self, gene_pathway: pd.DataFrame, relation_specs: Dict[str, Dict[str, object]]
    ):
        super().__init__()
        binary = torch.tensor(gene_pathway.to_numpy(), dtype=torch.float32)
        self.register_buffer(
            "gene_to_path", binary / binary.sum(0, keepdim=True).clamp_min(1.0)
        )
        transpose = binary.T
        self.register_buffer(
            "path_to_gene", transpose / transpose.sum(0, keepdim=True).clamp_min(1.0)
        )
        (n_genes, n_pathways) = binary.shape
        self.shared_encoder = encoder(n_pathways, config.HIDDEN_DIM, config.SHARED_DIM)
        self.shared_decoders = nn.ModuleDict(
            {
                name: decoder(config.SHARED_DIM, config.HIDDEN_DIM, n_pathways)
                for name in config.MODALITY_NAMES
            }
        )
        self.unique_encoders = nn.ModuleDict(
            {
                name: encoder(n_genes, config.HIDDEN_DIM, config.UNIQUE_DIM)
                for name in config.MODALITY_NAMES
            }
        )
        self.unique_decoders = nn.ModuleDict(
            {
                name: decoder(config.UNIQUE_DIM, config.HIDDEN_DIM, n_genes)
                for name in config.MODALITY_NAMES
            }
        )
        self.relation_gene_encoders = nn.ModuleDict()
        self.relation_pathway_encoders = nn.ModuleDict()
        for name in config.RELATION_NAMES:
            spec = relation_specs[name]
            matrix = torch.sparse_coo_tensor(
                torch.tensor(spec["indices"], dtype=torch.long),
                torch.tensor(spec["values"], dtype=torch.float32),
                size=tuple(spec["shape"]),
            ).coalesce()
            self.register_buffer(f"relation_{name}", matrix)
            self.register_buffer(
                f"mask_{name}", torch.tensor(spec["mask"], dtype=torch.float32)
            )
            self.relation_gene_encoders[name] = nn.Sequential(
                nn.Linear(2, config.SYNERGY_GENE_DIM),
                nn.GELU(),
                nn.Linear(config.SYNERGY_GENE_DIM, 1),
            )
            self.relation_pathway_encoders[name] = encoder(
                n_pathways, config.HIDDEN_DIM, config.SYNERGY_DIM
            )
        self.global_gene_encoder = nn.Sequential(
            nn.Linear(2, config.SYNERGY_GENE_DIM),
            nn.GELU(),
            nn.Linear(config.SYNERGY_GENE_DIM, 1),
        )
        self.global_pathway_encoder = encoder(
            n_pathways, config.HIDDEN_DIM, config.SYNERGY_DIM
        )
        self.synergy_merge = encoder(
            config.SYNERGY_DIM * 2, config.HIDDEN_DIM, config.SYNERGY_DIM
        )
        self.synergy_aux_head = nn.Linear(config.SYNERGY_DIM, 1)
        fusion_dim = (
            config.SHARED_DIM
            + len(config.MODALITY_NAMES) * config.UNIQUE_DIM
            + config.SYNERGY_DIM
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.CLASSIFIER_DIM),
            nn.LayerNorm(config.CLASSIFIER_DIM),
            nn.GELU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_DIM, config.CLASSIFIER_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.CLASSIFIER_DROPOUT),
            nn.Linear(config.CLASSIFIER_DIM // 2, 1),
        )
        self.register_buffer(
            "relation_quality",
            torch.stack(
                [
                    torch.tensor(relation_specs[name]["quality"], dtype=torch.float32)
                    for name in config.RELATION_NAMES
                ]
            ),
        )
        self.register_buffer(
            "relation_mask",
            torch.stack(
                [getattr(self, f"mask_{name}") for name in config.RELATION_NAMES]
            ),
        )
        self.relation_weight_net = nn.Sequential(
            nn.Linear(3, config.RELATION_WEIGHT_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(config.RELATION_WEIGHT_HIDDEN_DIM, 1),
        )
        nn.init.zeros_(self.relation_weight_net[-1].weight)
        nn.init.zeros_(self.relation_weight_net[-1].bias)
        self.relation_base_coefficients = nn.Parameter(
            torch.tensor(config.RELATION_BASE_COEFFICIENTS, dtype=torch.float32)
        )

    @staticmethod
    def sparse_project(source: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(matrix.transpose(0, 1).coalesce(), source.T).T

    @staticmethod
    def sign_preserving_rms(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        active = mask.unsqueeze(0)
        denominator = active.sum(dim=1, keepdim=True).clamp_min(1.0)
        rms = torch.sqrt(
            (values.pow(2) * active).sum(dim=1, keepdim=True) / denominator + 1e-06
        )
        return values / rms

    def encode_ru(
        self, mrna: torch.Tensor, mirna_gene: torch.Tensor, cnv: torch.Tensor
    ) -> Dict[str, object]:
        inputs = {"mrna": mrna, "mirna": mirna_gene, "cnv": cnv}
        pathways = {
            name: values @ self.gene_to_path for (name, values) in inputs.items()
        }
        shared_candidates = {
            name: self.shared_encoder(values) for (name, values) in pathways.items()
        }
        z_shared = torch.stack(
            [shared_candidates[name] for name in config.MODALITY_NAMES], dim=1
        ).mean(dim=1)
        (shared_pathway, shared_gene, z_unique, recon_gene) = ({}, {}, {}, {})
        for (name, values) in inputs.items():
            shared_pathway[name] = self.shared_decoders[name](z_shared)
            shared_gene[name] = shared_pathway[name] @ self.path_to_gene
            residual = values - shared_gene[name]
            z_unique[name] = self.unique_encoders[name](residual)
            recon_gene[name] = shared_gene[name] + self.unique_decoders[name](
                z_unique[name]
            )
        return {
            "inputs": inputs,
            "pathways": pathways,
            "shared_candidates": shared_candidates,
            "shared_pathway": shared_pathway,
            "shared_gene": shared_gene,
            "z_shared": z_shared,
            "z_unique": z_unique,
            "recon_gene": recon_gene,
        }

    def encode_synergy(
        self, mrna: torch.Tensor, mirna_raw: torch.Tensor, cnv: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        sources = {"cnv": cnv, "mirna": mirna_raw, "tf": mrna, "ctx": mrna}
        target_signal = torch.tanh(mrna)
        (agreements, relation_representations) = ({}, {})
        (positive_signals, conflict_signals) = ({}, {})
        for name in config.RELATION_NAMES:
            mask = getattr(self, f"mask_{name}")
            raw_pressure = self.sparse_project(
                sources[name], getattr(self, f"relation_{name}")
            )
            pressure = torch.tanh(
                self.sign_preserving_rms(raw_pressure, mask)
            ) * mask.unsqueeze(0)
            agreement = pressure * target_signal
            agreements[name] = agreement
            positive = F.relu(agreement)
            positive_signals[name] = positive
            conflict_signal = F.relu(-agreement - config.BIOLOGICAL_CONFLICT_MARGIN)
            conflict_signals[name] = conflict_signal * mask.unsqueeze(0)
            empty = torch.zeros_like(agreement)
            agreement_channels = torch.stack([positive, empty], dim=-1)
            agreement_gene = self.relation_gene_encoders[name](
                agreement_channels
            ).squeeze(-1)
            relation_gene = agreement_gene * mask.unsqueeze(0)
            relation_pathway = relation_gene @ self.gene_to_path
            relation_representations[name] = self.relation_pathway_encoders[name](
                relation_pathway
            )
        mask = self.relation_mask.unsqueeze(0)
        quality = self.relation_quality.unsqueeze(0)
        active_count = mask.sum(dim=2).clamp_min(1.0)
        positive_stack = torch.stack(
            [positive_signals[name] for name in config.RELATION_NAMES], dim=1
        )
        conflict_stack = torch.stack(
            [conflict_signals[name] for name in config.RELATION_NAMES], dim=1
        )
        weight_magnitude = positive_stack + conflict_stack
        weight_positive = positive_stack
        contribution_quality = quality.clamp_min(1e-06).pow(
            config.CONTRIBUTION_QUALITY_POWER
        )
        contribution = (weight_magnitude * contribution_quality * mask).sum(
            dim=2
        ) / active_count
        rho = contribution / contribution.sum(dim=1, keepdim=True).clamp_min(1e-06)
        quality_score = (quality * mask).sum(dim=2) / active_count
        quality_score = quality_score.expand_as(rho)
        direction_strength = weight_positive.sum(dim=2) / weight_magnitude.sum(
            dim=2
        ).clamp_min(1e-06)
        conflict_mass = (conflict_stack * mask).sum(dim=2) / active_count
        conflict_ratio = conflict_mass / (
            (positive_stack * mask).sum(dim=2) / active_count + conflict_mass
        ).clamp_min(1e-06)
        features = torch.stack([rho, quality_score, direction_strength], dim=-1)
        feature_mean = features.mean(dim=1, keepdim=True)
        feature_std = features.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-06)
        normalized = (features - feature_mean) / feature_std
        relation_logits = self.relation_weight_net(normalized).squeeze(-1)
        relation_logits = relation_logits + (
            normalized * self.relation_base_coefficients.view(1, 1, -1)
        ).sum(dim=-1)
        soft_weights = torch.softmax(relation_logits, dim=1)
        relation_lambda = (
            config.RELATION_WEIGHT_FLOOR
            + (1.0 - config.RELATION_WEIGHT_FLOOR * len(config.RELATION_NAMES))
            * soft_weights
        )
        calibrated_positive = positive_stack * relation_lambda.unsqueeze(-1)
        global_positive = calibrated_positive.sum(dim=1)
        empty_global = torch.zeros_like(global_positive)
        agreement_global_channels = torch.stack([global_positive, empty_global], dim=-1)
        agreement_global_gene = self.global_gene_encoder(
            agreement_global_channels
        ).squeeze(-1)
        global_z = self.global_pathway_encoder(
            agreement_global_gene @ self.gene_to_path
        )
        relation_z_stack = torch.stack(
            [relation_representations[name] for name in config.RELATION_NAMES], dim=1
        )
        relation_z = (relation_z_stack * relation_lambda.unsqueeze(-1)).sum(dim=1)
        z_synergy = self.synergy_merge(torch.cat([global_z, relation_z], dim=1))
        biological_conflict_features = torch.cat([conflict_ratio, conflict_mass], dim=1)
        return (
            z_synergy,
            {
                "agreements": agreements,
                "lambda": relation_lambda,
                "rho": rho,
                "quality": quality_score,
                "direction": direction_strength,
                "biological_conflict": biological_conflict_features,
            },
        )

    def forward(
        self,
        mrna: torch.Tensor,
        mirna_gene: torch.Tensor,
        cnv: torch.Tensor,
        mirna_raw: torch.Tensor,
    ) -> Dict[str, object]:
        ru = self.encode_ru(mrna, mirna_gene, cnv)
        factors = [ru["z_shared"]]
        factors.extend((ru["z_unique"][name] for name in config.MODALITY_NAMES))
        (z_synergy, relation) = self.encode_synergy(mrna, mirna_raw, cnv)
        factors.append(z_synergy)
        synergy_aux_logit = self.synergy_aux_head(z_synergy).squeeze(-1)
        logits = self.classifier(torch.cat(factors, dim=1)).squeeze(-1)
        return {
            "ru": ru,
            "z_synergy": z_synergy,
            "agreements": relation["agreements"],
            "relation": relation,
            "synergy_aux_logit": synergy_aux_logit,
            "logits": logits,
        }
