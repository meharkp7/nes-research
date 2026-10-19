import json, sys, numpy as np
sys.path.insert(0, "/workspace/nes_run_20261016_anchor_ablation/code/nes-llm/scripts")
sys.path.insert(0, "/workspace/nes_run_20261016_anchor_ablation/code/nes-llm")
from exp_detectability import MODELS, SIZES, SEEDS, BLOCKSIZE, METHOD, payload_for, key_for, features_for_blocks
from run_nf4_anchor_pilot import load_tensor
from src.experiments.nf4_anchor_channel import embed, unpack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

X, y, g = [], [], []
for model, spec in MODELS.items():
    for tensor in spec["tensors"]:
        src, st = load_tensor(spec["path"], tensor)
        raw = src.numpy().tobytes(); codes = unpack(raw)
        cb = st.code.float().numpy(); nb = len(codes) // BLOCKSIZE
        cl = codes[:nb * BLOCKSIZE].reshape(nb, BLOCKSIZE)
        for size in SIZES:
            for seed in SEEDS:
                stego, _ = embed(raw, payload_for(size, seed), key_for(seed), tensor, METHOD, cb)
                sc = unpack(stego); diff = sc != codes
                touch = diff.reshape(nb, BLOCKSIZE).any(1)
                if touch.sum() == 0:
                    continue
                f = features_for_blocks(sc.reshape(nb, BLOCKSIZE)[touch], cb)
                X.append(f); y.append(np.ones(len(f))); g.append(np.array([f"{model}::{tensor}"] * len(f)))
                cfc = cl[~touch]
                if len(cfc):
                    fc = features_for_blocks(cfc, cb)
                    X.append(fc); y.append(np.zeros(len(fc))); g.append(np.array([f"{model}::{tensor}"] * len(fc)))
X = np.concatenate(X); y = np.concatenate(y); g = np.concatenate(g)
oof = np.zeros(len(y))
for gg in sorted(set(g)):
    te = g == gg; tr = ~te
    sc = StandardScaler().fit(X[tr])
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=20261016)
    clf.fit(sc.transform(X[tr]), y[tr])
    oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
auc = float(roc_auc_score(y, oof))
fpr, tpr, _ = roc_curve(y, oof)
def tpr_at(t):
    i = np.searchsorted(fpr, t, side="right") - 1
    return float(tpr[max(i, 0)])
res = {"grouped_oracle_auc": auc, "tpr_at_fpr_0.01": tpr_at(0.01), "tpr_at_fpr_0.05": tpr_at(0.05),
       "n_examples": int(len(y)), "n_groups": int(len(set(g))),
       "note": "leave-one-tensor-out; modified blocks vs untouched blocks of same tensor"}
json.dump(res, open("/workspace/nes_run_20261016_anchor_ablation/out/detectability_supplement_v1/grouped_oracle.json", "w"), indent=2)
print(json.dumps(res))
