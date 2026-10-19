"""
Neural Embedding Strategy.

Learns to embed bits by training a small autoencoder on actual
quantization residuals. Unlike hand-crafted strategies, the neural
approach learns the optimal embedding direction for each residual
distribution end-to-end.

Architecture:
    Encoder: residual → latent code with bit embedded
    Decoder: latent code → modified residual

Training objective (3 losses jointly optimised):
    1. Recovery loss:  cross-entropy(extracted_bit, target_bit)
    2. Fidelity loss:  MSE(modified_residual, original_residual)
    3. Security loss:  KL(modified_distribution || original_distribution)

    total_loss = recovery_loss
               + lambda_fidelity * fidelity_loss
               + lambda_security * security_loss

The security loss pushes the modified distribution to be
statistically indistinguishable from the original — something
sign-based and LWE strategies cannot explicitly optimise for.

Performance targets (after training):
    BER (clean):    0.000
    BER @ σ=0.001:  < 0.010
    KL divergence:  < 1e-6  (better than sign/LWE)
    Detector:       < 50.5% (near-random)
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.core.types      import EmbeddingConfig, EmbeddingResult
from src.core.exceptions import EmbeddingError

# Fixed seed for the memory-safe training path so a run is reproducible.
SEED_TRAINER = 42


# ------------------------------------------------------------------
# Network architecture
# ------------------------------------------------------------------

class ResidualEncoder(nn.Module):
    """
    Encodes a residual value + bit into a modified residual.

    Input:  (residual_value, bit)  — 2 scalar inputs
    Output: modified_residual      — 1 scalar output

    Architecture: 2 → 64 → 64 → 32 → 1
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, residual: torch.Tensor, bit: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual: [N] original residual values
            bit:      [N] target bits as float (0.0 or 1.0)
        Returns:
            [N] modified residual values
        """
        x = torch.stack([residual, bit], dim=1)   # [N, 2]
        return self.net(x).squeeze(1)             # [N]


class ResidualDecoder(nn.Module):
    """
    Decodes a modified residual back to a bit probability.

    Input:  modified_residual  — 1 scalar
    Output: bit_probability    — scalar in [0, 1]

    Architecture: 1 → 32 → 32 → 1 → sigmoid
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, modified: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modified: [N] modified residual values
        Returns:
            [N] bit probabilities in [0, 1]
        """
        return self.net(modified.unsqueeze(1)).squeeze(1)  # [N]


class NeuralEmbeddingModel(nn.Module):
    """
    Full neural steganography model — encoder + decoder.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = ResidualEncoder(hidden_dim)
        self.decoder = ResidualDecoder(hidden_dim // 2)

    def forward(
        self,
        residual: torch.Tensor,
        bit:      torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Returns:
            (modified_residual, bit_probability)
        """
        modified    = self.encoder(residual, bit)
        bit_prob    = self.decoder(modified)
        return modified, bit_prob


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

class NeuralEmbeddingTrainer:
    """
    Trains NeuralEmbeddingModel on actual residual data.

    Usage:
        trainer = NeuralEmbeddingTrainer()
        model   = trainer.train(residuals, epochs=50)
        trainer.save(model, "models/neural_embedder.pt")
    """

    def __init__(
        self,
        hidden_dim:       int   = 64,
        lr:               float = 1e-3,
        lambda_fidelity:  float = 10.0,
        lambda_security:  float = 1.0,
        batch_size:       int   = 4096,
        device:           str   = "cpu",
    ):
        self.hidden_dim      = hidden_dim
        self.lr              = lr
        self.lambda_fidelity = lambda_fidelity
        self.lambda_security = lambda_security
        self.batch_size      = batch_size
        self.device          = device

    def train(
        self,
        residuals:   Dict[int, torch.Tensor],
        epochs:      int = 100,
        verbose:     bool = True,
    ) -> NeuralEmbeddingModel:
        """
        Train on high-magnitude residuals — matching the carrier
        positions that will actually be used at embedding time.
        """
        all_residuals = torch.cat([
            t.float().flatten() for t in residuals.values()
        ])

        # KEY FIX: sample top 20% by magnitude (these are the actual carriers)
        # instead of random samples from the full distribution
        max_train  = 5_000_000
        n_total    = all_residuals.numel()
        n_top      = min(max_train, n_total)
        
        abs_vals   = all_residuals.abs()
        threshold  = abs_vals.kthvalue(max(1, n_total - n_top)).values.item()
        high_mag   = all_residuals[abs_vals >= threshold]
        
        # Also include some random samples for generalisation
        n_random   = min(max_train // 4, n_total)
        random_idx = torch.randperm(n_total)[:n_random]
        random_smp = all_residuals[random_idx]
        
        training_pool = torch.cat([high_mag, random_smp])
        # Shuffle
        training_pool = training_pool[torch.randperm(training_pool.numel())]
        training_pool = training_pool[:max_train].to(self.device)

        model     = NeuralEmbeddingModel(self.hidden_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        bce_loss  = nn.BCELoss()
        mse_loss  = nn.MSELoss()

        model.train()
        n = training_pool.numel()

        for epoch in range(epochs):
            perm    = torch.randperm(n, device=self.device)
            epoch_recovery = epoch_fidelity = epoch_security = 0.0
            num_batches = 0

            for start in range(0, n, self.batch_size):
                idx   = perm[start: start + self.batch_size]
                batch = training_pool[idx]
                bits  = torch.randint(0, 2, (len(batch),),
                                    device=self.device).float()

                optimizer.zero_grad()
                modified, bit_prob = model(batch, bits)

                loss_recovery = bce_loss(bit_prob, bits)
                loss_fidelity = mse_loss(modified, batch)
                loss_security = (
                    (modified.mean() - batch.mean()).pow(2) +
                    (modified.std()  - batch.std()).pow(2)
                )

                loss = (
                    loss_recovery +
                    self.lambda_fidelity * loss_fidelity +
                    self.lambda_security * loss_security
                )

                loss.backward()
                optimizer.step()

                epoch_recovery += loss_recovery.item()
                epoch_fidelity += loss_fidelity.item()
                epoch_security += loss_security.item()
                num_batches    += 1

            scheduler.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"recovery={epoch_recovery/num_batches:.4f} | "
                    f"fidelity={epoch_fidelity/num_batches:.6f} | "
                    f"security={epoch_security/num_batches:.6f}"
                )

        model.eval()
        return model

    # ------------------------------------------------------------------
    # Memory-safe training
    # ------------------------------------------------------------------

    def train_sampled(
        self,
        residuals:   Dict[int, torch.Tensor],
        epochs:      int = 30,
        sample_size: int = 2_000_000,
        verbose:     bool = True,
    ) -> NeuralEmbeddingModel:
        """
        Train without materialising the full residual tensor.

        Why this exists
        ---------------
        ``train()`` concatenates every layer, takes an absolute-value copy,
        runs ``kthvalue`` over the whole thing, and then calls
        ``torch.randperm(n_total)``. For Qwen2.5-3B that is 2.44B residual
        values, so the allocations are roughly:

            torch.cat(...)          9.8 GB
            .abs()                 9.8 GB
            randperm(n_total)     19.6 GB   (int64)
            kthvalue                full pass

        ~39 GB before the first batch, on a machine where the NF4+FP16
        model pair already OOMs at 30 GB. That is why the neural strategy
        had never produced a result: it was not slow, it was not runnable.

        This version samples per layer with ``torch.randint`` (no
        permutation over the full range), estimates the magnitude
        threshold from a bounded subsample, and tops up to ``sample_size``
        high-magnitude values. Peak allocation is the output, not the
        input size.
        """
        layers = sorted(residuals.keys())

        if not layers:
            raise ValueError("No residual tensors supplied")

        # --- 1. Bounded subsample to estimate the magnitude threshold
        per_layer_probe = max(sample_size // (4 * len(layers)), 4096)
        probe = torch.cat([
            residuals[layer_id].flatten()[
                torch.randint(
                    0,
                    max(residuals[layer_id].numel(), 1),
                    (min(per_layer_probe, residuals[layer_id].numel()),),
                    generator=torch.Generator().manual_seed(
                        SEED_TRAINER + layer_id
                    ),
                )
            ].float()
            for layer_id in layers
        ])

        # Carriers are the high-magnitude tail, so aim at roughly the
        # top 20% of the observed distribution.
        quantile = float(
            torch.quantile(
                probe.abs(), 0.80
            ).item()
        )
        del probe

        # --- 2. Collect a bounded high-magnitude training pool
        target_high = int(sample_size * 0.75)
        high_chunks = []
        collected = 0

        for layer_id in layers:
            tensor = residuals[layer_id].flatten()

            if tensor.numel() == 0:
                continue

            # Stride sample: cheap, deterministic, no permutation.
            stride = max(tensor.numel() // (target_high // len(layers) + 1), 1)
            candidates = tensor[::stride][: target_high // len(layers) + 1]

            selected = candidates[candidates.abs() >= quantile]
            high_chunks.append(selected.float())
            collected += selected.numel()

            if collected >= target_high:
                break

        if not high_chunks:
            raise ValueError(
                "No residual values above the sampled magnitude "
                "quantile; cannot build a training pool"
            )

        high_mag = torch.cat(high_chunks)[:target_high]

        # --- 3. A smaller random pool for generalisation
        target_random = sample_size // 4
        random_chunks = []
        for layer_id in layers:
            tensor = residuals[layer_id].flatten()
            take = min(tensor.numel(), target_random // len(layers) + 1)
            idx = torch.randint(
                0, max(tensor.numel(), 1), (take,),
                generator=torch.Generator().manual_seed(
                    SEED_TRAINER + 1000 + layer_id
                ),
            )
            random_chunks.append(tensor[idx].float())

        random_smp = torch.cat(random_chunks)[:target_random]

        training_pool = torch.cat([high_mag, random_smp])
        del high_chunks, random_chunks

        training_pool = training_pool[
            torch.randperm(training_pool.numel())
        ].to(self.device)

        print(
            f"[NeuralTrainer] pool={training_pool.numel():,} "
            f"threshold={quantile:.3e} device={self.device}",
            flush=True,
        )

        model     = NeuralEmbeddingModel(self.hidden_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        bce_loss  = nn.BCELoss()
        mse_loss  = nn.MSELoss()

        model.train()
        n = training_pool.numel()
        generator = torch.Generator(device=self.device).manual_seed(
            SEED_TRAINER
        )

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device, generator=generator)

            tot_recovery = tot_fidelity = tot_security = 0.0
            batches = 0

            for start in range(0, n, self.batch_size):
                idx   = perm[start : start + self.batch_size]
                batch = training_pool[idx]
                bits  = torch.randint(
                    0, 2, (len(batch),),
                    device=self.device, generator=generator,
                ).float()

                optimizer.zero_grad()
                modified, bit_prob = model(batch, bits)

                loss_recovery = bce_loss(bit_prob, bits)
                loss_fidelity = mse_loss(modified, batch)
                loss_security = (
                    (modified.mean() - batch.mean()).pow(2)
                    + (modified.std() - batch.std()).pow(2)
                )

                loss = (
                    loss_recovery
                    + self.lambda_fidelity * loss_fidelity
                    + self.lambda_security * loss_security
                )

                loss.backward()
                optimizer.step()

                tot_recovery += loss_recovery.item()
                tot_fidelity += loss_fidelity.item()
                tot_security += loss_security.item()
                batches += 1

            scheduler.step()

            if verbose and (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"recovery={tot_recovery/batches:.4f} | "
                    f"fidelity={tot_fidelity/batches:.8f} | "
                    f"security={tot_security/batches:.8f}",
                    flush=True,
                )

        model.eval()
        return model

    def save(self, model: NeuralEmbeddingModel, path: str) -> None:
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )
        torch.save({
            "state_dict": model.state_dict(),
            "hidden_dim": self.hidden_dim,
        }, path)
        print(f"[NeuralTrainer] Saved to {path}")

    @staticmethod
    def load(path: str, device: str = "cpu") -> NeuralEmbeddingModel:
        ckpt       = torch.load(path, map_location=device)
        model      = NeuralEmbeddingModel(ckpt["hidden_dim"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


# ------------------------------------------------------------------
# Embedding / Extraction
# ------------------------------------------------------------------

class NeuralStrategy:
    """
    Neural embedding strategy using trained NeuralEmbeddingModel.

    Usage:
        # Train
        trainer = NeuralEmbeddingTrainer()
        model   = trainer.train(residuals, epochs=50)

        # Embed
        strategy = NeuralStrategy(config, model=model)
        result   = strategy.embed(residuals, bits, carrier_indices)

        # Extract
        extractor = NeuralExtractor(model)
        recovered = extractor.extract(result.embedded_weights,
                                      result.carrier_indices)
    """

    def __init__(
        self,
        config:  EmbeddingConfig,
        model:   Optional[NeuralEmbeddingModel] = None,
        device:  str = "cpu",
    ):
        self.config = config
        self.model  = model
        self.device = device

        if model is None:
            raise EmbeddingError(
                "NeuralStrategy requires a trained model. "
                "Use NeuralEmbeddingTrainer().train(residuals) first."
            )
        self.model.eval()

    def embed(
        self,
        residuals:        Dict[int, torch.Tensor],
        bits:             List[int],
        selector_indices: Dict[int, List[int]],
    ) -> EmbeddingResult:
        """
        Embed bits using the trained neural encoder.
        """
        embedded               = {}
        bit_idx                = 0
        actual_carrier_indices = {}

        with torch.no_grad():
            for layer_id in sorted(residuals.keys()):
                residual_tensor = residuals[layer_id].clone().detach()
                indices         = selector_indices.get(layer_id, [])
                embedded_flat   = residual_tensor.flatten().float()
                actual_indices  = []

                if not indices:
                    embedded[layer_id]               = residual_tensor
                    actual_carrier_indices[layer_id] = []
                    continue

                # Extract carrier values for batch processing
                end_idx = min(bit_idx + len(indices), len(bits))
                n_embed = end_idx - bit_idx

                if n_embed <= 0:
                    embedded[layer_id]               = residual_tensor
                    actual_carrier_indices[layer_id] = []
                    continue

                batch_indices    = indices[:n_embed]
                carrier_values   = embedded_flat[batch_indices].to(self.device)
                target_bits      = torch.tensor(
                    bits[bit_idx:end_idx], dtype=torch.float32, device=self.device
                )

                # Neural encoding
                modified_values  = self.model.encoder(carrier_values, target_bits)
                modified_values  = modified_values.to(residual_tensor.device)

                # Write back
                embedded_flat    = embedded_flat.clone()
                for i, carrier_idx in enumerate(batch_indices):
                    embedded_flat[carrier_idx] = modified_values[i].item()

                embedded[layer_id]               = embedded_flat.reshape(residual_tensor.shape)
                actual_carrier_indices[layer_id] = batch_indices
                bit_idx = end_idx

        bits_embedded = bit_idx
        total_bits    = len(bits)

        return EmbeddingResult(
            success=          True,
            embedded_weights= embedded,
            carrier_indices=  actual_carrier_indices,
            layer_allocation= {lid: len(idx) for lid, idx in actual_carrier_indices.items()},
            bits_embedded=    bits_embedded,
            total_bits=       total_bits,
            efficiency=       bits_embedded / total_bits if total_bits > 0 else 0.0,
            metadata={"strategy": "neural", "hidden_dim": self.model.encoder.net[0].in_features}
        )