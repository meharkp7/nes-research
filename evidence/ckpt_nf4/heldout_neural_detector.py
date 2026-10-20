#!/usr/bin/env python3
"""Held-out checkpoint neural detector (leave-one-model-out).

Trains MLP on 3 models' detectability features, tests on held-out 4th.
This validates whether neural detection generalizes across model architectures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add repo to path for imports
REPO = "/Users/meharkapoor7/nes-agent-workspace/nes-research/code/nes-llm"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/scripts")

from scripts.run_nf4_anchor_pilot import load_tensor, sha_file
from src.experiments.nf4_anchor_channel import embed, unpack
from scripts.exp_detectability import (
    FEATURES, BLOCKSIZE, METHOD, payload_for, key_for, features_for_blocks,
    write_json, cohen_d,
)

SEEDS = (20261016, 20261017, 20261018)
SIZES = (10000, 50000)
BLOCKS_PER_ARTIFACT = 2048
SAMPLE_SEED = 424242

# Model configurations with their checkpoints and tensors
MODELS = {
    "smollm2": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/smollm2_clean",
        "tensors": [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.11.self_attn.k_proj.weight",
            "model.layers.23.self_attn.v_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.11.mlp.down_proj.weight",
            "model.layers.23.mlp.gate_proj.weight",
        ],
    },
    "qwen15": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/qwen15_clean",
        "tensors": [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.13.self_attn.k_proj.weight",
            "model.layers.27.self_attn.v_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.13.mlp.down_proj.weight",
            "model.layers.27.mlp.gate_proj.weight",
        ],
    },
    "pythia": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/pythia_clean",
        "tensors": [
            "gpt_neox.layers.0.attention.query_key_value.weight",
            "gpt_neox.layers.11.attention.dense.weight",
            "gpt_neox.layers.23.attention.query_key_value.weight",
            "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
            "gpt_neox.layers.11.mlp.dense_4h_to_h.weight",
            "gpt_neox.layers.23.mlp.dense_h_to_4h.weight",
        ],
    },
    "falcon": {
        "checkpoint": "/Users/meharkapoor7/nes-agent-workspace/nes-research/evidence/ckpt_nf4/falcon_clean",
        "tensors": [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.8.self_attn.k_proj.weight",
            "model.layers.17.self_attn.v_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.8.mlp.down_proj.weight",
            "model.layers.17.mlp.gate_proj.weight",
        ],
    },
}

SEED = 20261016
TEST_RATIO = 0.20
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
HIDDEN_1 = 256
HIDDEN_2 = 64


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha_file(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def build_model_dataset(model_name, model_spec):
    """Build detectability dataset for a single model."""
    X_list, meta = [], []
    source_hashes = {}
    
    for tensor in model_spec["tensors"]:
        source, state = load_tensor(model_spec["checkpoint"], tensor)
        raw = source.numpy().tobytes()
        codes = unpack(raw)
        cb = state.code.float().numpy()
        nb = len(codes) // BLOCKSIZE
        clean_blocks = codes[:nb * BLOCKSIZE].reshape(nb, BLOCKSIZE)
        src_sha = hashlib.sha256(raw).hexdigest()
        source_hashes[tensor] = src_sha
        rng = np.random.default_rng(SAMPLE_SEED)
        idx = np.sort(rng.choice(nb, size=min(BLOCKS_PER_ARTIFACT, nb), replace=False))

        def add(blocks_all, m):
            f = features_for_blocks(blocks_all[idx], cb)
            X_list.append(f)
            meta.extend([{**m, "source_tensor_sha256": src_sha, "model": model_name}] * len(f))

        add(clean_blocks, {"tensor": tensor, "label": "clean", "size": 0, "seed": 0})
        for size in SIZES:
            for seed in SEEDS:
                stego, _ = embed(raw, payload_for(size, seed), key_for(seed), tensor, METHOD, cb)
                scodes = unpack(stego)
                add(scodes[:nb * BLOCKSIZE].reshape(nb, BLOCKSIZE),
                    {"tensor": tensor, "label": "embedded", "size": size, "seed": seed})
    
    X = np.concatenate(X_list)
    return X, meta, source_hashes


class ResidualDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class Detector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    tp = tn = fp = fn = 0
    all_probs = []
    all_labels = []
    
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()
        
        correct += (predictions == labels).sum().item()
        total += labels.numel()
        tp += ((predictions == 1) & (labels == 1)).sum().item()
        tn += ((predictions == 0) & (labels == 0)).sum().item()
        fp += ((predictions == 1) & (labels == 0)).sum().item()
        fn += ((predictions == 0) & (labels == 1)).sum().item()
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / max(total, 1)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "probs": all_probs,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Build datasets for all models
    print("Building datasets for all models...")
    all_X, all_meta = [], []
    all_source_hashes = {}
    
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}", file=sys.stderr)
            continue
        print(f"  Building {model_name}...")
        X, meta, src_hashes = build_model_dataset(model_name, MODELS[model_name])
        all_X.append(X)
        all_meta.extend(meta)
        all_source_hashes.update({f"{model_name}:{k}": v for k, v in src_hashes.items()})
        print(f"    {model_name}: {len(X)} samples, {len(meta)} meta entries")

    X = np.concatenate(all_X)
    label = np.array([1 if m["label"] == "embedded" else 0 for m in all_meta])
    model_arr = np.array([m["model"] for m in all_meta])
    tensor = np.array([m["tensor"] for m in all_meta])
    group = np.array([f"{m['model']}::{m['tensor']}" for m in all_meta])

    print(f"Total samples: {len(X)}")
    print(f"Models: {sorted(set(model_arr))}")
    print(f"Tensors per model: {len(set(tensor[model_arr == args.models[0]]))}")

    # Manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": f"nes_heldout_neural_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "scope": "Leave-one-model-out neural detector on 4 checkpoints",
        "models": args.models,
        "tensors_per_model": {m: MODELS[m]["tensors"] for m in args.models},
        "sizes": list(SIZES),
        "seeds": list(SEEDS),
        "blocks_per_artifact": BLOCKS_PER_ARTIFACT,
        "sample_seed": SAMPLE_SEED,
        "features": list(FEATURES),
        "detector_config": {
            "hidden_1": HIDDEN_1, "hidden_2": HIDDEN_2,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.lr, "dropout": [0.30, 0.20],
        },
        "source_hashes": all_source_hashes,
        "codec_sha256": sha_file(Path(f"{REPO}/src/experiments/nf4_anchor_channel.py")),
        "platform": platform.platform(),
        "versions": {m: importlib.metadata.version(m)
                     for m in ["numpy", "scikit-learn", "torch", "bitsandbytes", "safetensors"]},
    }
    write_json(args.output / "manifest.json", manifest)

    # Leave-one-model-out evaluation
    results = {"leave_one_model_out": {}}
    
    for held_out in args.models:
        print(f"\n{'='*60}")
        print(f"Held-out model: {held_out}")
        print(f"{'='*60}")
        
        # Train/test split by model
        train_mask = model_arr != held_out
        test_mask = model_arr == held_out
        
        X_train, y_train = X[train_mask], label[train_mask]
        X_test, y_test = X[test_mask], label[test_mask]
        
        print(f"Train samples: {len(X_train)} ({np.sum(y_train)} embedded)")
        print(f"Test samples:  {len(X_test)} ({np.sum(y_test)} embedded)")
        
        # Clean-vs-clean control on held-out model
        cl_test = np.where((test_mask) & (label == 0))[0]
        if len(cl_test) > 10:
            cX, cy = [], []
            for g in sorted(set(group[cl_test])):
                i = cl_test[group[cl_test] == g]
                h = len(i) // 2
                if h > 0:
                    cX += [X[i[:h]], X[i[h:2*h]]]
                    cy += [np.zeros(h), np.ones(h)]
            if cX:
                cX = np.concatenate(cX); cy = np.concatenate(cy)
                # Quick control evaluation
                pass  # We'll evaluate with the trained model
        
        # Standardize features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_n = (X_train - mean) / std
        X_test_n = (X_test - mean) / std
        
        # Train MLP
        train_dataset = ResidualDataset(X_train_n, y_train)
        test_dataset = ResidualDataset(X_test_n, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = Detector(input_dim=X_train.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        print("Training...")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                metrics = evaluate(model, test_loader, device)
                print(f"  Epoch {epoch+1:02d}/{args.epochs} loss={total_loss/len(train_loader):.6f} "
                      f"test_acc={metrics['accuracy']:.4f} test_auc={metrics['auc']:.4f}")
        
        # Final evaluation
        metrics = evaluate(model, test_loader, device)
        
        # Clean-vs-clean control on held-out test set
        cl_test = np.where((test_mask) & (label == 0))[0]
        ctrl_auc = None
        if len(cl_test) > 10:
            cX_list, cy_list = [], []
            for g in sorted(set(group[cl_test])):
                i = cl_test[group[cl_test] == g]
                h = len(i) // 2
                if h > 0:
                    cX_list += [X[i[:h]], X[i[h:2*h]]]
                    cy_list += [np.zeros(h), np.ones(h)]
            if cX_list:
                cX = np.concatenate(cX_list)
                cy = np.concatenate(cy_list)
                cX_n = (cX - mean) / std
                ctrl_dataset = ResidualDataset(cX_n, cy)
                ctrl_loader = DataLoader(ctrl_dataset, batch_size=args.batch_size, shuffle=False)
                ctrl_metrics = evaluate(model, ctrl_loader, device)
                ctrl_auc = ctrl_metrics["auc"]
                print(f"  Clean-vs-clean control AUC: {ctrl_auc:.4f}")
        
        results["leave_one_model_out"][held_out] = {
            "test_auc": metrics["auc"],
            "test_accuracy": metrics["accuracy"],
            "tp": metrics["tp"], "tn": metrics["tn"],
            "fp": metrics["fp"], "fn": metrics["fn"],
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "n_embedded_train": int(np.sum(y_train)),
            "n_embedded_test": int(np.sum(y_test)),
            "clean_vs_clean_auc": ctrl_auc,
        }
        print(f"  Final: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Summary
    aucs = [results["leave_one_model_out"][m]["test_auc"] for m in args.models]
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for m in args.models:
        r = results["leave_one_model_out"][m]
        print(f"  {m}: AUC={r['test_auc']:.4f}, Acc={r['test_accuracy']:.4f}")
    print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    results["summary"] = {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "individual_aucs": {m: results["leave_one_model_out"][m]["test_auc"] for m in args.models},
    }
    
    write_json(args.output / "heldout_neural_results.json", results)
    print(json.dumps({"status": "COMPLETED", **results["summary"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())