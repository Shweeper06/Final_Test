# -*- coding: utf-8 -*-
"""
Streamlit Application: German Credit Risk — Model, Explainability & Fairness
แอปนี้ "ไม่ฝึกโมเดลใหม่" แต่โหลดผลที่ฝึกและปรับไว้แล้วจาก Google Colab
  - german_credit.csv       : ข้อมูลดิบ (สำหรับหน้า EDA)
  - model_artifacts.joblib  : Pipeline ที่ปรับแล้ว, threshold, ผล CV, ผลทดสอบ, permutation importance
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="German Credit Risk App", layout="wide")


# ---------------------------------------------------------------------------
# 1) โหลดไฟล์ที่ส่งออกจาก Colab (cache ไว้เพื่อไม่ให้โหลดซ้ำทุกครั้งที่กดปุ่ม)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("german_credit.csv")


@st.cache_resource
def load_artifacts() -> dict:
    return joblib.load("model_artifacts.joblib")


# ---------------------------------------------------------------------------
# 2) ฟังก์ชัน fairness (เหมือนกับใน Colab)
# ---------------------------------------------------------------------------
def group_metrics(y_true, y_pred, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).values
        yt, yp = y_true[m], y_pred[m]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append({
            "group": g,
            "n": int(m.sum()),
            "actual_bad_rate": yt.mean(),
            "pred_bad_rate": yp.mean(),
            "approval_rate": 1 - yp.mean(),
            "TPR": tp / (tp + fn) if (tp + fn) else np.nan,
            "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
            "precision_bad": tp / (tp + fp) if (tp + fp) else np.nan,
        })
    return pd.DataFrame(rows)


def fairness_summary(tbl: pd.DataFrame) -> dict:
    ar, tpr, fpr = tbl["approval_rate"], tbl["TPR"], tbl["FPR"]
    return {
        "Demographic parity diff.": ar.max() - ar.min(),
        "Disparate impact ratio": ar.min() / ar.max() if ar.max() > 0 else np.nan,
        "Equal opportunity diff. (TPR)": tpr.max() - tpr.min(),
        "FPR diff.": fpr.max() - fpr.min(),
        "Equalized odds diff.": max(tpr.max() - tpr.min(), fpr.max() - fpr.min()),
    }


# ---------------------------------------------------------------------------
# 3) เริ่มต้นแอป
# ---------------------------------------------------------------------------
try:
    raw = load_data()
    art = load_artifacts()
except FileNotFoundError as e:
    st.error(f"ไม่พบไฟล์ที่ต้องใช้: {e.filename} — กรุณาอัปโหลดไฟล์จาก Colab ขึ้น GitHub ให้ครบ")
    st.stop()

meta = art["meta"]
results = art["results"]
best_name = meta["best_model"]
COST_FN, COST_FP = meta["cost_fn"], meta["cost_fp"]

data = raw.copy()
data["sex"] = np.where(data["personal_status"].str.startswith("female"), "female", "male")
data["age_group"] = np.where(data["age"] < 25, "<25", ">=25")

st.title("German Credit Risk: Model, Explainability & Fairness")
st.caption(
    "ชุดข้อมูล Statlog German Credit (OpenML id=31) · positive class = 'bad' · "
    f"ต้นทุน FN:FP = {COST_FN}:{COST_FP} · โมเดลฝึกและปรับใน Google Colab"
)

with st.sidebar:
    st.header("การตั้งค่า")
    model_name = st.selectbox(
        "โมเดลที่ใช้วิเคราะห์",
        list(results.keys()),
        index=list(results.keys()).index(best_name),
        help=f"ค่าเริ่มต้นคือโมเดลที่ CV AUC สูงสุด ({best_name})",
    )
    res = results[model_name]
    st.metric("Threshold (จาก OOF)", f"{res['threshold']:.2f}")
    st.caption(f"scikit-learn ที่ใช้ฝึก: {meta['versions']['scikit-learn']}")

tabs = st.tabs([
    "1. ข้อมูลและ EDA", "2. โมเดลและ CV", "3. Threshold ตามต้นทุน",
    "4. Permutation Importance", "5. Fairness", "6. ทดลองทำนาย",
])

# ---------------- แท็บ 1: EDA ----------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("จำนวนแถว", f"{raw.shape[0]:,}")
    c2.metric("จำนวนฟีเจอร์", raw.shape[1] - 1)
    c3.metric("สัดส่วน 'bad'", f"{(raw['class'] == 'bad').mean():.1%}")

    st.subheader("ตัวอย่างข้อมูล")
    st.dataframe(raw.head(20), width="stretch")

    cc1, cc2 = st.columns([1, 2])
    cc1.markdown("**ค่าที่หายไป**")
    cc1.dataframe(raw.isna().sum().rename("missing"), width="stretch")
    cc2.markdown("**สถิติเชิงพรรณนา**")
    cc2.dataframe(raw.describe().T, width="stretch")

    num_cols = raw.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in raw.columns if c not in num_cols + ["class"]]

    st.subheader("ตัวแปรเชิงตัวเลขเทียบกับ target")
    num_sel = st.selectbox("เลือกตัวแปรตัวเลข", num_cols,
                           index=num_cols.index("credit_amount") if "credit_amount" in num_cols else 0)
    st.plotly_chart(px.box(data, x="class", y=num_sel, color="class"), width="stretch")

    st.subheader("อัตรา 'bad' ในแต่ละหมวด")
    cat_sel = st.selectbox("เลือกตัวแปรหมวดหมู่", cat_cols + ["sex", "age_group"])
    rate = (data.groupby(cat_sel)["class"].apply(lambda s: (s == "bad").mean())
            .rename("bad_rate").reset_index())
    st.plotly_chart(px.bar(rate, x=cat_sel, y="bad_rate", text_auto=".1%"), width="stretch")

    st.subheader("Correlation ของตัวแปรตัวเลข")
    st.plotly_chart(px.imshow(raw[num_cols].corr(), text_auto=".2f",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                    width="stretch")

# ---------------- แท็บ 2: โมเดลและ CV ----------------
with tabs[1]:
    st.markdown(
        f"แบ่ง train/test แบบ stratified (random_state={meta['random_state']}, "
        f"test_size={meta['test_size']}) · ปรับ hyperparameter ด้วย GridSearchCV + "
        f"StratifiedKFold (k={meta['n_splits']}) บนชุดฝึกเท่านั้น"
    )
    summary = pd.DataFrame([
        {"model": n, "CV AUC": r["cv_auc"], "OOF AUC": r["oof_auc"],
         "threshold": r["threshold"], **r["test_metrics"]}
        for n, r in results.items()
    ]).set_index("model")
    st.dataframe(summary.style.format(precision=3), width="stretch")
    st.info(f"โมเดลที่ CV AUC สูงสุด: **{best_name}**")
    st.subheader(f"รายละเอียด: {model_name}")
    st.write("Best hyperparameters:", res["best_params"])
    st.dataframe(res["cv_table"], width="stretch")

# ---------------- แท็บ 3: Threshold ----------------
with tabs[2]:
    st.markdown(f"Expected cost = **{COST_FN} × FN + {COST_FP} × FP** จาก out-of-fold predictions บนชุดฝึก")
    fig = px.line(res["cost_curve"], x="threshold", y="cost", title="Expected cost (OOF)")
    fig.add_vline(x=res["threshold"], line_dash="dash",
                  annotation_text=f"best = {res['threshold']:.2f}")
    st.plotly_chart(fig, width="stretch")

    cm = confusion_matrix(art["y_test"], res["pred_test"], labels=[0, 1])
    st.subheader("Confusion matrix บนชุดทดสอบ")
    st.dataframe(pd.DataFrame(cm, index=["จริง: good", "จริง: bad"],
                              columns=["ทำนาย: good", "ทำนาย: bad"]), width="content")
    m = res["test_metrics"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Cost @ threshold ที่เลือก", int(m["Cost @ best t"]))
    k2.metric("Cost @ 0.5", int(m["Cost @ 0.5"]))
    k3.metric("Recall (bad)", f"{m['Recall (bad)']:.2f}")

# ---------------- แท็บ 4: Permutation importance ----------------
with tabs[3]:
    imp = res["perm_importance"]
    st.plotly_chart(
        px.bar(imp.head(15).iloc[::-1], x="importance_mean", y="feature",
               error_x="importance_std", orientation="h",
               title="AUC ที่ลดลงเมื่อสลับค่าฟีเจอร์ (ชุดทดสอบ)"),
        width="stretch",
    )
    st.dataframe(imp, width="stretch")

# ---------------- แท็บ 5: Fairness ----------------
with tabs[4]:
    st.markdown("วัดบนชุดทดสอบ · ผลลัพธ์ที่เป็นประโยชน์ต่อผู้กู้คือ **ได้รับอนุมัติ (ทำนายว่า good)**")
    y_test = np.asarray(art["y_test"])
    for attr in ["sex", "age_group"]:
        st.subheader(f"ตาม {attr}")
        tbl = group_metrics(y_test, res["pred_test"], art["groups_test"][attr])
        st.dataframe(tbl.style.format(precision=3), width="stretch")
        fs = fairness_summary(tbl)
        for col, (k, v) in zip(st.columns(len(fs)), fs.items()):
            col.metric(k, f"{v:.3f}")
        di = fs["Disparate impact ratio"]
        (st.warning if di < 0.8 else st.success)(
            f"Disparate impact = {di:.2f} ({'ต่ำกว่า' if di < 0.8 else 'ผ่าน'}เกณฑ์ 0.8)"
        )
    if "fairness_compare" in art:
        st.subheader("เปรียบเทียบ: ตัด personal_status และ age ออกจากฟีเจอร์ (คำนวณใน Colab)")
        st.dataframe(art["fairness_compare"].style.format(precision=3), width="stretch")

# ---------------- แท็บ 6: ทดลองทำนาย ----------------
with tabs[5]:
    X_ref = art["X_train"]
    default = art["X_test"].iloc[0]
    with st.form("predict_form"):
        inputs = {}
        grid = st.columns(3)
        for i, c in enumerate(X_ref.columns):
            box = grid[i % 3]
            if pd.api.types.is_numeric_dtype(X_ref[c]):
                inputs[c] = box.number_input(c, value=float(default[c]))
            else:
                opts = sorted(X_ref[c].astype(str).unique().tolist())
                inputs[c] = box.selectbox(c, opts, index=opts.index(str(default[c])))
        submitted = st.form_submit_button("ทำนาย")
    if submitted:
        row = pd.DataFrame([inputs])[X_ref.columns]
        p = res["pipeline"].predict_proba(row)[0, 1]
        decision = "bad (ไม่อนุมัติ)" if p >= res["threshold"] else "good (อนุมัติ)"
        st.metric("ความน่าจะเป็นที่จะเป็น 'bad'", f"{p:.1%}")
        st.write(f"ผลการตัดสินที่ threshold {res['threshold']:.2f}: **{decision}**")
