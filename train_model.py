# -*- coding: utf-8 -*-
"""
train_model.py — ฝึกและประเมิน 4 โมเดลสำหรับ German Credit Risk
ใช้ได้ 3 แบบ
  1) ใน Google Colab:  !python train_model.py        (สร้าง german_credit.csv + model_artifacts.joblib)
  2) บนเครื่อง:        python train_model.py
  3) ใน app.py:        ถ้าไม่พบ / โหลด model_artifacts.joblib ไม่ได้ แอปจะเรียก build_artifacts() เอง

โครงสร้างผลลัพธ์ (artifacts) ตรงกับที่ Notebook บันทึก เพื่อให้ app.py ใช้ได้ทั้งสองทาง
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
COST_FN, COST_FP = 5, 1          # FN = ปล่อยกู้ให้คนที่ bad, FP = ปฏิเสธคนที่ good
POS_LABEL = "bad"
SENSITIVE = ["personal_status", "age"]
DATA_FILE = "german_credit.csv"
ARTIFACT_FILE = "model_artifacts.joblib"


# --------------------------------------------------------------------------- data
def load_raw(csv_path: str = DATA_FILE) -> pd.DataFrame:
    """อ่านจาก CSV ถ้ามี มิฉะนั้นดึงจาก OpenML (data_id=31)"""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=31, as_frame=True).frame
    for c in df.select_dtypes(include="category").columns:
        df[c] = df[c].astype(str)
    return df


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sex"] = np.where(out["personal_status"].str.startswith("female"), "female", "male")
    out["age_group"] = np.where(out["age"] < 25, "<25", ">=25")
    return out


# --------------------------------------------------------------------------- model
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def get_model_specs() -> dict:
    """4 อัลกอริทึม พร้อม grid ของ hyperparameter"""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1.0, 10.0]}),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
            {"model__n_estimators": [200, 400], "model__max_depth": [None, 8],
             "model__min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [2, 3]}),
        "Support Vector Machine": (
            SVC(kernel="rbf", probability=True, class_weight="balanced",
                random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.01]}),
    }


# --------------------------------------------------------------------------- metrics
def cost_of(y_true, pred, cost_fn=COST_FN, cost_fp=COST_FP) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return int(cost_fn * fn + cost_fp * fp)


def find_best_threshold(y_true, proba, cost_fn=COST_FN, cost_fp=COST_FP):
    rows = []
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({"threshold": t, "FN": int(fn), "FP": int(fp),
                     "cost": int(cost_fn * fn + cost_fp * fp)})
    curve = pd.DataFrame(rows)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"]), curve


def test_metrics(y_true, proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, proba),
        "Recall (bad)": recall_score(y_true, pred, zero_division=0),
        "Precision (bad)": precision_score(y_true, pred, zero_division=0),
        "F1 (bad)": f1_score(y_true, pred, zero_division=0),
        "Cost @ best t": cost_of(y_true, pred, cost_fn, cost_fp),
        "Cost @ 0.5": cost_of(y_true, (proba >= 0.5).astype(int), cost_fn, cost_fp),
    }


def group_metrics(y_true, y_pred, groups: pd.Series) -> pd.DataFrame:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).values
        yt, yp = y_true[m], y_pred[m]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({"group": g, "n": int(m.sum()),
                     "actual_bad_rate": yt.mean(), "pred_bad_rate": yp.mean(),
                     "approval_rate": 1 - yp.mean(),
                     "TPR": tp / (tp + fn) if (tp + fn) else np.nan,
                     "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
                     "precision_bad": tp / (tp + fp) if (tp + fp) else np.nan})
    return pd.DataFrame(rows)


def fairness_summary(tbl: pd.DataFrame) -> dict:
    ar, tpr, fpr = tbl["approval_rate"], tbl["TPR"], tbl["FPR"]
    return {"Demographic parity diff.": ar.max() - ar.min(),
            "Disparate impact ratio": ar.min() / ar.max() if ar.max() > 0 else np.nan,
            "Equal opportunity diff. (TPR)": tpr.max() - tpr.min(),
            "FPR diff.": fpr.max() - fpr.min(),
            "Equalized odds diff.": max(tpr.max() - tpr.min(), fpr.max() - fpr.min())}


# --------------------------------------------------------------------------- pipeline
def build_artifacts(df: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """ฝึก 4 โมเดล → เลือก threshold จาก OOF → ประเมินบนชุดทดสอบ → importance → fairness"""
    log = print if verbose else (lambda *a, **k: None)
    df = add_groups(load_raw() if df is None else df)

    y = (df["class"] == POS_LABEL).astype(int)
    X = df.drop(columns=["class", "sex", "age_group"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    groups_test = df.loc[X_test.index, ["sex", "age_group"]]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, (est, grid) in get_model_specs().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", est)])
        gs = GridSearchCV(pipe, grid, cv=skf, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_

        oof = cross_val_predict(best_pipe, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
        t, curve = find_best_threshold(y_train.values, oof)
        proba = best_pipe.predict_proba(X_test)[:, 1]

        pi = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc",
                                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
        results[name] = {
            "pipeline": best_pipe,
            "best_params": gs.best_params_,
            "cv_auc": gs.best_score_,
            "cv_table": pd.DataFrame(gs.cv_results_)[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ].sort_values("rank_test_score").reset_index(drop=True),
            "oof_proba": oof,
            "oof_auc": roc_auc_score(y_train, oof),
            "threshold": t,
            "cost_curve": curve,
            "proba_test": proba,
            "pred_test": (proba >= t).astype(int),
            "test_metrics": test_metrics(y_test, proba, t),
            "perm_importance": pd.DataFrame({
                "feature": X_test.columns,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std,
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True),
        }
        log(f"{name:24s} CV AUC={gs.best_score_:.4f}  t={t:.2f}  "
            f"test AUC={results[name]['test_metrics']['AUC']:.4f}")

    best_name = max(results, key=lambda n: results[n]["cv_auc"])
    best = results[best_name]

    # fairness: รวม vs ตัดตัวแปรอ่อนไหว
    fair_rows = []
    for attr in ["sex", "age_group"]:
        fs = fairness_summary(group_metrics(y_test.values, best["pred_test"], groups_test[attr]))
        fair_rows.append({"attribute": attr, "features": "รวมตัวแปรอ่อนไหว", **fs})
    Xtr2, Xte2 = X_train.drop(columns=SENSITIVE), X_test.drop(columns=SENSITIVE)
    pipe2 = clone(best["pipeline"]).set_params(prep=build_preprocessor(Xtr2))
    pipe2.fit(Xtr2, y_train)
    oof2 = cross_val_predict(pipe2, Xtr2, y_train, cv=skf, method="predict_proba")[:, 1]
    t2, _ = find_best_threshold(y_train.values, oof2)
    pred2 = (pipe2.predict_proba(Xte2)[:, 1] >= t2).astype(int)
    for attr in ["sex", "age_group"]:
        fs = fairness_summary(group_metrics(y_test.values, pred2, groups_test[attr]))
        fair_rows.append({"attribute": attr, "features": "ตัดตัวแปรอ่อนไหว", **fs})
    fairness_compare = (pd.DataFrame(fair_rows)
                        .sort_values(["attribute", "features"]).reset_index(drop=True))

    log("โมเดลที่เลือก (CV AUC สูงสุด):", best_name)
    return {
        "meta": {"best_model": best_name, "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
                 "n_splits": N_SPLITS, "cost_fn": COST_FN, "cost_fp": COST_FP,
                 "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                              "numpy": np.__version__}},
        "results": results,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "groups_test": groups_test, "fairness_compare": fairness_compare,
    }


if __name__ == "__main__":
    raw = load_raw()
    raw.to_csv(DATA_FILE, index=False)
    art = build_artifacts(raw)
    joblib.dump(art, ARTIFACT_FILE, compress=3)
    print(f"บันทึก {DATA_FILE} และ {ARTIFACT_FILE} เรียบร้อย "
          f"({os.path.getsize(ARTIFACT_FILE) / 1024:,.0f} KB)")
