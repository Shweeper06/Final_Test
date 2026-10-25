"""
credit_core.py
==============
โมดูลหลักสำหรับงาน Statlog German Credit (OpenML data_id=31)
แยกส่วน "การคำนวณ" ออกจากส่วน "หน้าจอ" (app.py) เพื่อให้
  - อ่านและตรวจโค้ดได้ง่าย
  - ทดสอบแต่ละขั้นตอนได้โดยไม่ต้องเปิด Streamlit
  - app.py เรียกใช้ฟังก์ชันเหล่านี้แล้วนำผลไปแสดง

ลำดับขั้นตอนในไฟล์นี้ตรงกับข้อสอบ
  ข้อ 1   load_data(), add_group_columns()         โหลดข้อมูล + เตรียมตัวแปรกลุ่ม
  ข้อ 2.1 make_xy(), split_data()                  กำหนด target และแยก train/test แบบ stratified
  ข้อ 2.2 build_preprocessor(), model_catalog()    Pipeline = ColumnTransformer + โมเดล (4 อัลกอริทึม)
  ข้อ 2.3 tune_model()                              GridSearchCV + StratifiedKFold(k=5) บนชุดฝึกเท่านั้น
  ข้อ 2.4 oof_proba(), find_best_threshold()       หา threshold ที่ลด expected cost (FN:FP = 5:1)
  ข้อ 3   permutation_importance_table(), shap_importance_table()
  ข้อ 4   fairness_table(), fairness_summary()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------
# ค่าคงที่ของงาน
# ----------------------------------------------------------------------------
RANDOM_STATE = 42          # ตามโจทย์ข้อ 2.1
TEST_SIZE = 0.20           # สัดส่วนชุดทดสอบ 20%
N_SPLITS = 5               # k = 5 ตามโจทย์ข้อ 2.3
COST_FN = 5                # ต้นทุนของ False Negative (ทายว่า good แต่จริงเป็น bad)
COST_FP = 1                # ต้นทุนของ False Positive (ทายว่า bad แต่จริงเป็น good)
POS_LABEL = "bad"          # positive class ตามโจทย์ข้อ 2.1
TARGET_COL = "class"

BACKUP_CSV = Path(__file__).parent / "data" / "credit_g_backup.csv"


# ============================================================================
# ข้อ 1  โหลดข้อมูล
# ============================================================================
def load_data() -> tuple[pd.DataFrame, str]:
    """โหลดชุดข้อมูลจาก OpenML (data_id=31) ตามคำสั่งในโจทย์

    ถ้าเครื่องเชื่อมต่อ OpenML ไม่ได้ (เช่น เครือข่ายถูกบล็อก) จะอ่านไฟล์สำรอง
    data/credit_g_backup.csv ซึ่งมีชื่อคอลัมน์และค่าหมวดหมู่ตรงกับ OpenML
    คืนค่า (DataFrame, แหล่งข้อมูลที่ใช้จริง)
    """
    try:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=31, as_frame=True).frame
        source = "OpenML (data_id=31)"
    except Exception:  # noqa: BLE001 - ต้องการ fallback ทุกกรณีที่โหลดไม่สำเร็จ
        df = pd.read_csv(BACKUP_CSV)
        source = "ไฟล์สำรอง data/credit_g_backup.csv (โหลดจาก OpenML ไม่สำเร็จ)"

    # ทำชนิดข้อมูลให้สม่ำเสมอ: ตัวแปรข้อความเป็น category, ตัวเลขเป็น float
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(float)
        else:
            df[c] = df[c].astype(str).astype("category")
    return df, source


def add_group_columns(df: pd.DataFrame, age_cut: int = 25) -> pd.DataFrame:
    """สร้างตัวแปรกลุ่มสำหรับวิเคราะห์ความเป็นธรรม (ข้อ 4)

    - sex       : ข้อมูลชุดนี้ไม่มีคอลัมน์เพศโดยตรง จึงแยกจาก personal_status
                  (ค่าที่ขึ้นต้นด้วย 'female' = female, 'male' = male)
    - age_group : แบ่งอายุเป็น 2 กลุ่มที่จุดตัด age_cut (ค่าเริ่มต้น 25 ปี
                  ซึ่งเป็นจุดตัดที่งานวิจัยด้าน fairness ของชุดข้อมูลนี้นิยมใช้)
    คอลัมน์ทั้งสองใช้ "จัดกลุ่ม" เท่านั้น ไม่ได้ป้อนเข้าโมเดล
    """
    out = df.copy()
    ps = out["personal_status"].astype(str)
    out["sex"] = np.where(ps.str.startswith("female"), "female", "male")
    out["age_group"] = np.where(out["age"] < age_cut, f"<{age_cut}", f">={age_cut}")
    return out


# ============================================================================
# ข้อ 2.1  กำหนด target และแยก train/test แบบ stratified
# ============================================================================
GROUP_COLS = ["sex", "age_group"]
SENSITIVE_FEATURES = ["personal_status", "age"]


def make_xy(df: pd.DataFrame, drop_sensitive: bool = False):
    """y = 1 เมื่อ class == 'bad' (positive class), y = 0 เมื่อ 'good'

    drop_sensitive=True จะตัด personal_status และ age ออกจาก X
    (ใช้ทดลองดูว่าการไม่ใส่ตัวแปรอ่อนไหวช่วยลดความเหลื่อมล้ำได้หรือไม่)
    """
    y = (df[TARGET_COL].astype(str) == POS_LABEL).astype(int)
    drop = [TARGET_COL] + GROUP_COLS
    if drop_sensitive:
        drop += SENSITIVE_FEATURES
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return X, y


def split_data(X, y):
    """แยกชุดฝึก/ทดสอบ 80/20 โดย stratify=y เพื่อรักษาสัดส่วน bad:good (30:70)
    ให้เท่ากันทั้งสองชุด และกำหนด random_state=42 เพื่อให้ผลซ้ำได้
    ตัวแปรกลุ่ม (sex, age_group) ดึงภายหลังด้วย X_test.index"""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# ============================================================================
# ข้อ 2.2  Pipeline = ColumnTransformer + โมเดล
# ============================================================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """ตัวแปรตัวเลข  -> StandardScaler (สำคัญกับ Logistic Regression และ SVM)
    ตัวแปรหมวดหมู่ -> OneHotEncoder(handle_unknown='ignore') กันหมวดใหม่ในชุดทดสอบ

    การใส่ขั้นตอนนี้ไว้ "ใน" Pipeline ทำให้ตอนทำ CV ตัว scaler/encoder
    ถูก fit เฉพาะข้อมูลฝึกของแต่ละ fold จึงไม่เกิด data leakage
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )


def model_catalog() -> dict:
    """อัลกอริทึม 4 แบบ พร้อมช่วง hyperparameter ที่จะค้นหา

    ชื่อพารามิเตอร์ขึ้นต้นด้วย 'model__' เพราะโมเดลเป็นขั้นตอนชื่อ 'model'
    ใน Pipeline (GridSearchCV จะส่งค่าไปยังขั้นตอนนั้นโดยตรง)
    ช่วงค่าถูกจำกัดให้เล็กพอที่จะรันบน Streamlit Community Cloud ได้ภายในไม่กี่นาที
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.5, 1.0, 5.0],
                "model__gamma": ["scale", 0.01],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(X)), ("model", estimator)])


# ============================================================================
# ข้อ 2.3  ปรับ hyperparameter ด้วย StratifiedKFold (k=5) เฉพาะบนชุดฝึก
# ============================================================================
def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def tune_model(name: str, X_train, y_train) -> GridSearchCV:
    """GridSearchCV ใช้ ROC-AUC เป็นเกณฑ์ (ไม่ขึ้นกับ threshold)
    เพราะ threshold จะถูกเลือกแยกต่างหากในข้อ 2.4
    ส่งเฉพาะ X_train, y_train เข้าไป ชุดทดสอบจึงไม่ถูกใช้ในขั้นนี้เลย
    """
    estimator, grid = model_catalog()[name]
    search = GridSearchCV(
        build_pipeline(X_train, estimator),
        param_grid=grid,
        scoring="roc_auc",
        cv=make_cv(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


# ============================================================================
# ข้อ 2.4  เลือก threshold ที่ลด expected cost จาก out-of-fold predictions
# ============================================================================
def oof_proba(best_pipeline: Pipeline, X_train, y_train) -> np.ndarray:
    """Out-of-fold (OOF) probability: แต่ละแถวของชุดฝึกได้ค่าทำนายจากโมเดล
    ที่ "ไม่เคยเห็น" แถวนั้นตอนฝึก จึงใช้เลือก threshold ได้โดยไม่แตะชุดทดสอบ"""
    return cross_val_predict(
        best_pipeline, X_train, y_train, cv=make_cv(), method="predict_proba", n_jobs=-1
    )[:, 1]


def expected_cost(y_true, proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP) -> float:
    """ต้นทุนเฉลี่ยต่อราย = (5*FN + 1*FP) / n
    ทำนาย bad (1) เมื่อ proba >= threshold"""
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    y_true = np.asarray(y_true)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return (cost_fn * fn + cost_fp * fp) / len(y_true)


def threshold_curve(y_true, proba, cost_fn=COST_FN, cost_fp=COST_FP) -> pd.DataFrame:
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    return pd.DataFrame(
        {"threshold": grid,
         "expected_cost": [expected_cost(y_true, proba, t, cost_fn, cost_fp) for t in grid]}
    )


def find_best_threshold(y_true, proba, cost_fn=COST_FN, cost_fp=COST_FP):
    """ไล่ threshold 0.01–0.99 แล้วเลือกจุดที่ expected cost ต่ำสุด
    (เชิงทฤษฎี ถ้าความน่าจะเป็นถูก calibrate ดี จุดที่เหมาะสมคือ
    cost_fp / (cost_fp + cost_fn) = 1/6 ≈ 0.167 ใช้เป็นค่าอ้างอิงเปรียบเทียบได้)"""
    curve = threshold_curve(y_true, proba, cost_fn, cost_fp)
    best = curve.loc[curve["expected_cost"].idxmin()]
    return float(best["threshold"]), float(best["expected_cost"]), curve


# ----------------------------------------------------------------------------
# ประเมินผลบนชุดทดสอบ
# ----------------------------------------------------------------------------
def classification_summary(y_true, proba, threshold) -> dict:
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "ROC-AUC": roc_auc_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (bad)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (bad)": recall_score(y_true, y_pred, zero_division=0),
        "F1 (bad)": f1_score(y_true, y_pred, zero_division=0),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Expected cost / ราย": expected_cost(y_true, proba, threshold),
        "Total cost": int(COST_FN * fn + COST_FP * fp),
    }


# ============================================================================
# ข้อ 3  การอธิบายโมเดลบนชุดทดสอบ
# ============================================================================
def permutation_importance_table(pipeline, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    """สลับค่าของตัวแปรทีละตัวในชุดทดสอบ แล้ววัดว่า ROC-AUC ลดลงเท่าใด
    ทำบน Pipeline ทั้งก้อน ผลจึงเป็นระดับ "ตัวแปรต้นฉบับ" (ก่อน one-hot)
    อ่านผลง่ายกว่าการดูทีละคอลัมน์ dummy"""
    r = permutation_importance(
        pipeline, X_test, y_test,
        scoring="roc_auc", n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame({"feature": X_test.columns,
                      "importance_mean": r.importances_mean,
                      "importance_std": r.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _output_to_original(prep: ColumnTransformer) -> list[str]:
    """จับคู่คอลัมน์หลัง one-hot กลับไปยังชื่อตัวแปรต้นฉบับ
    เพื่อรวมค่า SHAP ของ dummy ทุกตัวในตัวแปรเดียวกัน"""
    mapping: list[str] = []
    for name, trans, cols in prep.transformers_:
        if name == "num":
            mapping += list(cols)
        elif name == "cat":
            for col, cats in zip(cols, trans.categories_):
                mapping += [col] * len(cats)
    return mapping


def shap_importance_table(pipeline, X_train, X_test, max_eval_rows: int = 200):
    """คำนวณ SHAP บนชุดทดสอบ (ข้อ 3 ส่วน "และ/หรือ SHAP")

    - โมเดลต้นไม้ (RF, GB) ใช้ TreeExplainer ซึ่งเร็วและคำนวณแบบแม่นตรง
    - โมเดลอื่น (LR, SVM) ใช้ Permutation explainer โดยมี background
      จากชุดฝึก 100 แถว (ช้ากว่า จึงจำกัดจำนวนแถวที่อธิบาย)
    คืนค่า (ตารางค่าเฉลี่ย |SHAP| ต่อตัวแปรต้นฉบับ, DataFrame ค่า SHAP รายแถว)
    """
    import shap

    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    Z_train = prep.transform(X_train)
    X_eval = X_test.iloc[:max_eval_rows]
    Z_eval = prep.transform(X_eval)

    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Z_eval)
        if isinstance(sv, list):          # shap รุ่นเก่า: list ต่อ class
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # shap รุ่นใหม่: (n, features, classes)
            sv = sv[:, :, 1]
    else:
        background = shap.sample(Z_train, 100, random_state=RANDOM_STATE)
        f = lambda Z: model.predict_proba(Z)[:, 1]  # noqa: E731
        explainer = shap.Explainer(f, shap.maskers.Independent(background), algorithm="permutation")
        sv = explainer(Z_eval, max_evals=2 * Z_eval.shape[1] + 1).values

    # รวมค่า SHAP ของคอลัมน์ dummy กลับเป็นตัวแปรต้นฉบับ
    owners = _output_to_original(prep)
    shap_df = pd.DataFrame(sv, columns=owners).T.groupby(level=0).sum().T
    shap_df.index = X_eval.index
    table = (
        shap_df.abs().mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return table, shap_df


# ============================================================================
# ข้อ 4  Fairness metrics ตาม sex และ age_group (บนชุดทดสอบ)
# ============================================================================
def fairness_table(y_true, y_pred, group: pd.Series) -> pd.DataFrame:
    """ตัวชี้วัดรายกลุ่ม
    - selection_rate (bad) : สัดส่วนที่โมเดลทำนายว่า 'bad' (ถูกปฏิเสธ)
    - approval_rate (good) : 1 - selection_rate (ได้รับอนุมัติ)
    - base_rate            : สัดส่วน 'bad' จริงในกลุ่ม
    - TPR (recall)         : จับ bad ได้ถูกกี่ %
    - FPR                  : คน good ถูกทำนายผิดว่า bad กี่ % (ถูกปฏิเสธโดยไม่สมควร)
    - FNR                  : คน bad หลุดเป็น good กี่ %
    """
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(group)})
    rows = []
    for g, s in d.groupby("g"):
        pos, neg = s[s.y == 1], s[s.y == 0]
        sel = s.p.mean()
        rows.append({
            "group": g,
            "n": len(s),
            "base_rate (bad จริง)": s.y.mean(),
            "selection_rate (ทาย bad)": sel,
            "approval_rate (ทาย good)": 1 - sel,
            "TPR": pos.p.mean() if len(pos) else np.nan,
            "FPR": neg.p.mean() if len(neg) else np.nan,
            "FNR": 1 - pos.p.mean() if len(pos) else np.nan,
            "Accuracy": (s.y == s.p).mean(),
        })
    return pd.DataFrame(rows).set_index("group")


def fairness_summary(table: pd.DataFrame) -> dict:
    """ตัวชี้วัดความต่างระหว่างกลุ่ม (ค่า 0 = เท่าเทียม)
    - Demographic parity difference : ส่วนต่าง selection rate สูงสุด-ต่ำสุด
    - Disparate impact ratio        : approval rate ต่ำสุด / สูงสุด
                                       (กฎ four-fifths: ควร >= 0.80)
    - Equal opportunity difference  : ส่วนต่าง TPR
    - Equalized odds difference     : ค่ามากกว่าระหว่างส่วนต่าง TPR และ FPR
    """
    sel = table["selection_rate (ทาย bad)"]
    appr = table["approval_rate (ทาย good)"]
    tpr_gap = table["TPR"].max() - table["TPR"].min()
    fpr_gap = table["FPR"].max() - table["FPR"].min()
    return {
        "Demographic parity difference": float(sel.max() - sel.min()),
        "Disparate impact ratio (approval)": float(appr.min() / appr.max()) if appr.max() > 0 else np.nan,
        "Equal opportunity difference (TPR)": float(tpr_gap),
        "Equalized odds difference": float(max(tpr_gap, fpr_gap)),
    }
