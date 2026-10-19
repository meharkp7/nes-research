#!/usr/bin/env python3
"""Train first-pass packed-NF4 detectors on the existing frozen split.

Exploratory pilot only: one model per partition is not enough to establish
generalization. This script does not change or bypass the readiness gate.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = {"artifact_id","artifact_sha256","run_id","source_id","model_id",
        "tensor_key","role","label","block_index","split"}
SPLITS = ("train","validation","test")

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def score_metrics(y, scores, threshold):
    pred=(scores>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n":int(len(y)),
      "roc_auc":float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
      "accuracy":float(accuracy_score(y,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),
      "recall":float(recall_score(y,pred,zero_division=0)),
      "f1":float(f1_score(y,pred,zero_division=0)),
      "confusion_matrix_tn_fp_fn_tp":[int(tn),int(fp),int(fn),int(tp)],
      "false_positive_rate":float(fp/max(1,fp+tn)),"threshold":float(threshold)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261010)
    a=p.parse_args(); data=a.dataset.expanduser().resolve(); out=a.output_dir.expanduser().resolve()
    if not data.is_file(): raise FileNotFoundError(data)
    if out.exists(): raise FileExistsError(f"Refusing to overwrite: {out}")
    df=pd.read_csv(data)
    missing=sorted(META-set(df.columns))
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if df.empty: raise ValueError("Dataset is empty")
    expected=df["role"].map({"clean":"0","embedded":"1"})
    if expected.isna().any() or not expected.equals(df["label"].astype(str)):
        raise ValueError("Role/label mismatch or unknown role")
    features=[c for c in df.columns if c not in META]
    bad=[c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if not features or bad: raise ValueError(f"No numeric features or non-numeric features: {bad}")
    if not set(df["split"].unique()).issubset(SPLITS): raise ValueError("Unexpected split values")
    parts={s:df[df.split==s].copy() for s in SPLITS}
    for s,d in parts.items():
        if d.empty or set(d.label.astype(int).unique())!={0,1}:
            raise ValueError(f"{s} must contain both labels")
    groups={s:sorted(d.model_id.unique().tolist()) for s,d in parts.items()}
    if (set(groups["train"])&set(groups["validation"]) or
        set(groups["train"])&set(groups["test"]) or
        set(groups["validation"])&set(groups["test"])):
        raise ValueError("Model identity leakage across split partitions")
    X={s:parts[s][features].to_numpy(dtype=float) for s in SPLITS}
    y={s:parts[s].label.astype(int).to_numpy() for s in SPLITS}
    models={
      "logistic_regression":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
        LogisticRegression(C=1.0,max_iter=1000,class_weight="balanced",random_state=a.seed)),
      "hist_gradient_boosting":make_pipeline(SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(max_iter=100,learning_rate=0.1,max_leaf_nodes=15,
          l2_regularization=1.0,random_state=a.seed))}
    report={"schema":"nes.packed_nf4_detector_training_pilot.v1","status":"COMPLETED",
      "created_utc":datetime.now(timezone.utc).isoformat(),"dataset":str(data),
      "dataset_sha256":sha256_file(data),"rows":int(len(df)),"feature_columns":features,
      "metadata_excluded_from_features":sorted(META),
      "model_groups":groups,"split_rows":{s:int(len(parts[s])) for s in SPLITS},
      "models":{},"interpretation":{
        "classification":"Exploratory pilot only; current split has one model group per partition.",
        "generalization":"Not established. Blocks within artifacts/runs are correlated; no independent-sample confidence intervals.",
        "stealth":"These metrics do not establish stealth or undetectability.",
        "controls":"Clean-vs-clean and label-randomization controls are not run by this script."}}
    for name,model in models.items():
        model.fit(X["train"],y["train"])
        vs=model.predict_proba(X["validation"])[:,1]
        thresholds=np.unique(np.concatenate(([0.0,0.5,1.0],vs)))
        def j(t):
            pred=vs>=t
            tpr=float(np.mean(pred[y["validation"]==1]))
            fpr=float(np.mean(pred[y["validation"]==0]))
            return (tpr-fpr,-abs(float(t)-0.5))
        threshold=max(thresholds,key=j)
        ts=model.predict_proba(X["test"])[:,1]
        report["models"][name]={"validation":score_metrics(y["validation"],vs,threshold),
                                "test":score_metrics(y["test"],ts,threshold)}
    out.mkdir(parents=True,exist_ok=False)
    dest=out/"detector_training_report.json"
    dest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":"COMPLETED","report":str(dest),"split_rows":report["split_rows"],
      "model_groups":groups,"results":report["models"],"warning":report["interpretation"]["classification"]},indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
