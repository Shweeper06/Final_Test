# -*- coding: utf-8 -*-
"""
Streamlit Application: German Credit Risk — Model, Explainability & Fairness

- โหมดพื้นฐาน : กรอกข้อมูลสำคัญ 8 รายการ → ได้ผลประเมินความเสี่ยงทันที + สรุปผลโมเดล
- โหมดละเอียด : กรอกครบ 20 ตัวแปร, ปรับ threshold/ต้นทุน, ประเมินหลายรายจาก CSV,
                 EDA, ผล CV, ROC, Permutation importance และ Fairness

แอปโหลดผลที่ฝึกไว้จาก Colab (model_artifacts.joblib) หากไม่พบหรือโหลดไม่ได้
จะฝึกใหม่อัตโนมัติด้วย train_model.py (ใช้เวลาประมาณ 1 นาทีในครั้งแรก)
"""
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

import train_model as tm

st.set_page_config(page_title="German Credit Risk", page_icon="💳", layout="wide")

# =============================================================================
# ป้ายภาษาไทยของตัวแปรและค่าหมวดหมู่ (ค่าที่ไม่อยู่ใน dict จะแสดงค่าดิบ)
# =============================================================================
FEATURE_TH = {
    "checking_status": "สถานะบัญชีกระแสรายวัน (DM)",
    "duration": "ระยะเวลากู้ (เดือน)",
    "credit_history": "ประวัติสินเชื่อ",
    "purpose": "วัตถุประสงค์การกู้",
    "credit_amount": "วงเงินกู้ (DM)",
    "savings_status": "เงินออม (DM)",
    "employment": "อายุงานปัจจุบัน (ปี)",
    "installment_commitment": "ภาระผ่อนต่อรายได้ (1–4)",
    "personal_status": "เพศ/สถานภาพ",
    "other_parties": "ผู้กู้ร่วม/ผู้ค้ำ",
    "residence_since": "ระยะเวลาที่อยู่อาศัยปัจจุบัน (1–4)",
    "property_magnitude": "ทรัพย์สิน",
    "age": "อายุ (ปี)",
    "other_payment_plans": "แผนผ่อนชำระอื่น",
    "housing": "ที่อยู่อาศัย",
    "existing_credits": "จำนวนสินเชื่อที่มีอยู่",
    "job": "ลักษณะงาน",
    "num_dependents": "จำนวนผู้อยู่ในอุปการะ",
    "own_telephone": "มีโทรศัพท์ในชื่อตนเอง",
    "foreign_worker": "แรงงานต่างชาติ",
}
VALUE_TH = {
    "checking_status": {"<0": "ติดลบ (< 0)", "0<=X<200": "0 – 199", ">=200": "≥ 200",
                        "no checking": "ไม่มีบัญชี"},
    "credit_history": {"no credits/all paid": "ไม่เคยกู้ / ชำระครบทุกแห่ง",
                       "all paid": "ชำระครบ (ธนาคารนี้)", "existing paid": "ผ่อนปกติ",
                       "delayed previously": "เคยชำระล่าช้า",
                       "critical/other existing credit": "มีปัญหา / มีหนี้ที่อื่น"},
    "purpose": {"new car": "รถใหม่", "used car": "รถมือสอง",
                "furniture/equipment": "เฟอร์นิเจอร์/อุปกรณ์", "radio/tv": "วิทยุ/โทรทัศน์",
                "domestic appliance": "เครื่องใช้ในบ้าน", "repairs": "ซ่อมแซม",
                "education": "การศึกษา", "vacation": "ท่องเที่ยว", "retraining": "ฝึกอาชีพ",
                "business": "ธุรกิจ", "other": "อื่น ๆ"},
    "savings_status": {"<100": "< 100", "100<=X<500": "100 – 499", "500<=X<1000": "500 – 999",
                       ">=1000": "≥ 1,000", "no known savings": "ไม่มี / ไม่ทราบ"},
    "employment": {"unemployed": "ว่างงาน", "<1": "< 1 ปี", "1<=X<4": "1 – 3 ปี",
                   "4<=X<7": "4 – 6 ปี", ">=7": "≥ 7 ปี"},
    "personal_status": {"male div/sep": "ชาย หย่า/แยกกันอยู่",
                        "female div/dep/mar": "หญิง หย่า/แยก/สมรส",
                        "male single": "ชาย โสด", "male mar/wid": "ชาย สมรส/หม้าย",
                        "female single": "หญิง โสด"},
    "other_parties": {"none": "ไม่มี", "co applicant": "ผู้กู้ร่วม", "guarantor": "ผู้ค้ำประกัน"},
    "property_magnitude": {"real estate": "อสังหาริมทรัพย์", "life insurance": "ประกันชีวิต",
                           "car": "รถยนต์/อื่น ๆ", "no known property": "ไม่มี / ไม่ทราบ"},
    "other_payment_plans": {"bank": "ธนาคาร", "stores": "ร้านค้า", "none": "ไม่มี"},
    "housing": {"rent": "เช่า", "own": "เป็นเจ้าของ", "for free": "อยู่ฟรี", "free": "อยู่ฟรี"},
    "job": {"unemp/unskilled non res": "ว่างงาน/ไร้ทักษะ (ไม่ใช่ผู้พำนัก)",
            "unskilled resident": "ไร้ทักษะ (ผู้พำนัก)", "skilled": "มีทักษะ",
            "high qualif/self emp/mgmt": "ทักษะสูง/อาชีพอิสระ/ผู้บริหาร"},
    "own_telephone": {"none": "ไม่มี", "yes": "มี"},
    "foreign_worker": {"yes": "ใช่", "no": "ไม่ใช่"},
}
BASIC_FEATURES = ["checking_status", "duration", "credit_amount", "credit_history",
                  "purpose", "savings_status", "employment", "age"]
MODEL_TH = {"Logistic Regression": "การถดถอยโลจิสติก",
            "Random Forest": "ป่าสุ่ม",
            "Gradient Boosting": "Gradient Boosting",
            "Support Vector Machine": "ซัพพอร์ตเวกเตอร์แมชชีน"}


def ft(c: str) -> str:
    return FEATURE_TH.get(c, c)


def vt(c: str, v) -> str:
    return VALUE_TH.get(c, {}).get(str(v), str(v))


# =============================================================================
# โหลดข้อมูล / โมเดล
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return tm.load_raw(tm.DATA_FILE)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """คืนค่า (artifacts, แหล่งที่มา, ข้อความเตือน)"""
    note = ""
    if os.path.exists(tm.ARTIFACT_FILE):
        try:
            art = joblib.load(tm.ARTIFACT_FILE)
            saved = art["meta"]["versions"]["scikit-learn"]
            if saved != sklearn.__version__:
                raise ValueError(f"scikit-learn ไม่ตรงกัน (ไฟล์ {saved} / ระบบ {sklearn.__version__})")
            art["results"][art["meta"]["best_model"]]["pipeline"].predict_proba(
                art["X_test"].head(1))
            return art, "Colab (model_artifacts.joblib)", note
        except Exception as e:  # noqa: BLE001
            note = f"โหลด model_artifacts.joblib ไม่สำเร็จ: {e} — ฝึกโมเดลใหม่แทน"
    else:
        note = "ไม่พบ model_artifacts.joblib — ฝึกโมเดลใหม่บนเซิร์ฟเวอร์"
    return tm.build_artifacts(load_data(), verbose=False), "ฝึกใหม่บนเซิร์ฟเวอร์", note


with st.spinner("กำลังโหลดข้อมูลและโมเดล (ครั้งแรกอาจใช้เวลาประมาณ 1 นาที)..."):
    try:
        raw = load_data()
        art, source, load_note = load_artifacts()
    except Exception as e:  # noqa: BLE001
        st.error(f"ไม่สามารถโหลดข้อมูลหรือโมเดลได้: {e}")
        st.stop()

meta, results = art["meta"], art["results"]
best_name = meta["best_model"]
X_ref: pd.DataFrame = art["X_train"]
X_test: pd.DataFrame = art["X_test"]
y_test = np.asarray(art["y_test"])
y_train = np.asarray(art["y_train"]) if "y_train" in art else None
MODEL_NAMES = list(results.keys())
NUM_COLS = X_ref.select_dtypes(include="number").columns.tolist()
DEFAULTS = {c: (float(X_ref[c].median()) if c in NUM_COLS else X_ref[c].mode()[0])
            for c in X_ref.columns}

data = tm.add_groups(raw)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("การตั้งค่า")
    mode = st.radio("รูปแบบการใช้งาน", ["พื้นฐาน", "ละเอียด"], horizontal=True,
                    help="พื้นฐาน: กรอกข้อมูลสำคัญและดูผลทันที · ละเอียด: ปรับค่าทุกส่วนและดูผลวิเคราะห์ครบ")
    advanced = mode == "ละเอียด"

    model_name = st.selectbox(
        "โมเดล", MODEL_NAMES, index=MODEL_NAMES.index(best_name),
        format_func=lambda n: f"{n}{'  ★ แนะนำ' if n == best_name else ''}",
        help=f"★ = โมเดลที่ CV AUC สูงสุดบนชุดฝึก ({best_name})")
    res = results[model_name]

    cost_fn, cost_fp = meta["cost_fn"], meta["cost_fp"]
    thr = float(res["threshold"])
    if advanced:
        st.divider()
        st.subheader("Threshold การตัดสิน")
        thr_mode = st.radio("วิธีกำหนด", ["จาก OOF (ค่าที่ฝึกไว้)", "คำนวณจากต้นทุนใหม่", "กำหนดเอง"])
        if thr_mode == "คำนวณจากต้นทุนใหม่":
            if "oof_proba" in res and y_train is not None:
                c1, c2 = st.columns(2)
                cost_fn = c1.number_input("ต้นทุน FN", 1, 50, int(meta["cost_fn"]),
                                          help="อนุมัติให้ผู้ที่จะผิดนัดชำระ")
                cost_fp = c2.number_input("ต้นทุน FP", 1, 50, int(meta["cost_fp"]),
                                          help="ปฏิเสธผู้ที่ชำระได้ดี")
                thr, _ = tm.find_best_threshold(y_train, res["oof_proba"], cost_fn, cost_fp)
            else:
                st.info("ไฟล์โมเดลไม่มีค่า OOF — ใช้ Notebook/train_model.py รุ่นใหม่เพื่อเปิดใช้")
        elif thr_mode == "กำหนดเอง":
            thr = st.slider("Threshold", 0.01, 0.99, thr, 0.01)
    st.metric("Threshold ที่ใช้", f"{thr:.2f}",
              help="ความน่าจะเป็น 'bad' ตั้งแต่ค่านี้ขึ้นไป = ไม่อนุมัติ")
    st.caption(f"ที่มาของโมเดล: {source}\n\nscikit-learn {sklearn.__version__}")
    if load_note:
        st.caption(f"หมายเหตุ: {load_note}")

# ค่าที่ขึ้นกับ threshold ปัจจุบัน
proba_test = np.asarray(res["proba_test"])
pred_test = (proba_test >= thr).astype(int)
cur_metrics = tm.test_metrics(y_test, proba_test, thr, cost_fn, cost_fp)


# =============================================================================
# ฟังก์ชันส่วนแสดงผล
# =============================================================================
def predict_row(row: pd.DataFrame, name: str) -> float:
    return float(results[name]["pipeline"].predict_proba(row[X_ref.columns])[0, 1])


def input_widget(c, box, key_prefix, default):
    label = ft(c)
    if c in NUM_COLS:
        lo, hi = float(X_ref[c].min()), float(X_ref[c].max())
        is_int = np.allclose(X_ref[c] % 1, 0)
        if is_int:
            return box.number_input(label, int(lo), int(hi * 2), int(round(float(default))),
                                    step=1, key=f"{key_prefix}_{c}")
        return box.number_input(label, lo, hi * 2, float(default), key=f"{key_prefix}_{c}")
    opts = sorted(X_ref[c].astype(str).unique().tolist())
    idx = opts.index(str(default)) if str(default) in opts else 0
    return box.selectbox(label, opts, index=idx, format_func=lambda v, c=c: vt(c, v),
                         key=f"{key_prefix}_{c}")


def gauge(p: float, t: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "ความน่าจะเป็นที่จะผิดนัด (bad)"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#E45756" if p >= t else "#2E8B57"},
               "steps": [{"range": [0, t * 100], "color": "#E8F5E9"},
                         {"range": [t * 100, 100], "color": "#FDECEA"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": t * 100}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def local_explanation(row: pd.DataFrame, name: str) -> pd.DataFrame:
    """ผลต่างของความน่าจะเป็น เมื่อแทนค่าตัวแปรนั้นด้วยค่ากลางของชุดฝึก
    ค่าบวก = ค่าที่กรอกทำให้ความเสี่ยงสูงขึ้นเมื่อเทียบกับผู้กู้ทั่วไป"""
    base = predict_row(row, name)
    variants = pd.concat([row] * len(X_ref.columns), ignore_index=True)
    for i, c in enumerate(X_ref.columns):
        variants.loc[i, c] = DEFAULTS[c]
    p = results[name]["pipeline"].predict_proba(variants[X_ref.columns])[:, 1]
    out = pd.DataFrame({"feature": X_ref.columns, "effect": base - p})
    out["label"] = [f"{ft(c)}: {vt(c, row.iloc[0][c])}" for c in out["feature"]]
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index)


def show_result(row: pd.DataFrame, detailed: bool):
    p = predict_row(row, model_name)
    approved = p < thr
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(gauge(p, thr), width="stretch")
    with c2:
        st.markdown("#### ผลการประเมิน")
        if approved:
            st.success(f"**ความเสี่ยงต่ำ — แนะนำอนุมัติ**\n\nความน่าจะเป็น bad {p:.1%} "
                       f"ต่ำกว่าเกณฑ์ {thr:.2f}")
        else:
            st.error(f"**ความเสี่ยงสูง — แนะนำไม่อนุมัติ / พิจารณาเพิ่มเติม**\n\n"
                     f"ความน่าจะเป็น bad {p:.1%} ตั้งแต่เกณฑ์ {thr:.2f} ขึ้นไป")
        st.caption(f"โมเดล: {model_name} · เกณฑ์ตั้งจากต้นทุน FN:FP = "
                   f"{meta['cost_fn']}:{meta['cost_fp']} จึงต่ำกว่า 0.5")

    st.markdown("##### ผลจากทั้ง 4 โมเดล")
    comp = pd.DataFrame([{
        "โมเดล": n + (" ★" if n == best_name else ""),
        "ความน่าจะเป็น bad": predict_row(row, n),
        "threshold": results[n]["threshold"] if n != model_name else thr,
    } for n in MODEL_NAMES])
    comp["ผล"] = np.where(comp["ความน่าจะเป็น bad"] >= comp["threshold"], "ไม่อนุมัติ", "อนุมัติ")
    st.dataframe(comp.style.format({"ความน่าจะเป็น bad": "{:.1%}", "threshold": "{:.2f}"}),
                 width="stretch", hide_index=True)

    st.markdown("##### ปัจจัยที่มีผลต่อรายนี้มากที่สุด")
    ex = local_explanation(row, model_name).head(8 if not detailed else 20).iloc[::-1]
    fig = px.bar(ex, x="effect", y="label", orientation="h",
                 color=np.where(ex["effect"] > 0, "เพิ่มความเสี่ยง", "ลดความเสี่ยง"),
                 color_discrete_map={"เพิ่มความเสี่ยง": "#E45756", "ลดความเสี่ยง": "#2E8B57"},
                 labels={"effect": "ผลต่อความน่าจะเป็น bad", "label": "", "color": ""})
    fig.update_layout(height=max(300, 32 * len(ex)), xaxis_tickformat="+.0%")
    st.plotly_chart(fig, width="stretch")
    st.caption("วิธีคำนวณ: เปลี่ยนค่าตัวแปรทีละตัวเป็นค่ากลางของชุดฝึก แล้ววัดว่าความน่าจะเป็นเปลี่ยนไปเท่าใด "
               "ใช้เพื่อประกอบการอธิบายเบื้องต้น ไม่ใช่ความสัมพันธ์เชิงเหตุผล")


def summary_table() -> pd.DataFrame:
    rows = []
    for n, r in results.items():
        t = thr if n == model_name else float(r["threshold"])
        m = tm.test_metrics(y_test, np.asarray(r["proba_test"]), t, cost_fn, cost_fp)
        rows.append({"model": n, "CV AUC": r["cv_auc"], "OOF AUC": r.get("oof_auc", np.nan),
                     "threshold": t, **m})
    return pd.DataFrame(rows).set_index("model")


# =============================================================================
# ส่วนหัว
# =============================================================================
st.title("German Credit Risk")
st.caption("ประเมินความเสี่ยงสินเชื่อด้วย 4 โมเดล · ชุดข้อมูล Statlog German Credit (OpenML id=31) · "
           "positive class = 'bad'")

# =============================================================================
# โหมดพื้นฐาน
# =============================================================================
if not advanced:
    tab_pred, tab_sum, tab_about = st.tabs(["ประเมินความเสี่ยง", "สรุปผลโมเดล", "เกี่ยวกับแอป"])

    with tab_pred:
        st.markdown("กรอกข้อมูลสำคัญ 8 รายการ ตัวแปรอื่นจะใช้ค่าที่พบบ่อยที่สุดในชุดฝึกโดยอัตโนมัติ "
                    "(เปลี่ยนเป็นโหมด **ละเอียด** ที่แถบด้านซ้ายเพื่อกรอกครบทุกตัวแปร)")
        with st.form("basic_form"):
            vals = {}
            cols = st.columns(4)
            for i, c in enumerate(BASIC_FEATURES):
                vals[c] = input_widget(c, cols[i % 4], "b", DEFAULTS[c])
            go_btn = st.form_submit_button("ประเมินความเสี่ยง", type="primary", width="stretch")
        if go_btn:
            row = pd.DataFrame([{**DEFAULTS, **vals}])[X_ref.columns]
            show_result(row, detailed=False)

    with tab_sum:
        s = summary_table()
        st.markdown("ผลบน **ชุดทดสอบ** (ข้อมูลที่โมเดลไม่เคยเห็น 20%)")
        k = st.columns(4)
        for col, n in zip(k, MODEL_NAMES):
            col.metric(n + (" ★" if n == best_name else ""), f"AUC {s.loc[n, 'AUC']:.3f}",
                       f"Recall {s.loc[n, 'Recall (bad)']:.2f}", delta_color="off")
        fig = px.bar(s.reset_index(), x="model", y=["CV AUC", "AUC"], barmode="group",
                     labels={"value": "ROC-AUC", "variable": "", "model": ""})
        fig.update_yaxes(range=[0.5, 1])
        st.plotly_chart(fig, width="stretch")
        st.info(f"โมเดลแนะนำคือ **{best_name}** เพราะ CV AUC บนชุดฝึกสูงสุด "
                "(เลือกจากชุดฝึก ไม่ได้เลือกจากชุดทดสอบ)")

    with tab_about:
        st.markdown(f"""
**วัตถุประสงค์** ประเมินว่าผู้ขอสินเชื่อมีแนวโน้มผิดนัดชำระ (bad) หรือไม่

**วิธีการโดยสรุป**
1. แบ่งข้อมูล train/test 80:20 แบบ stratified
2. ฝึก 4 โมเดล: {", ".join(MODEL_NAMES)} และปรับ hyperparameter ด้วย 5-fold CV
3. เลือก threshold ที่ทำให้ต้นทุนต่ำสุด (FN มีต้นทุน {meta['cost_fn']} เท่าของ FP)
4. ตรวจความสำคัญของตัวแปรและความเป็นธรรมตามเพศและอายุ

**ข้อควรระวัง** ผลนี้เป็นงานวิเคราะห์เชิงการศึกษา ไม่ควรใช้ตัดสินสินเชื่อจริงโดยลำพัง
""")
        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนรายการ", f"{len(raw):,}")
        c2.metric("จำนวนตัวแปร", raw.shape[1] - 1)
        c3.metric("สัดส่วน bad", f"{(raw['class'] == 'bad').mean():.1%}")

# =============================================================================
# โหมดละเอียด
# =============================================================================
else:
    tabs = st.tabs(["ประเมินรายบุคคล", "ประเมินหลายราย (CSV)", "ข้อมูลและ EDA", "โมเดลและ CV",
                    "Threshold และต้นทุน", "Permutation Importance", "Fairness"])

    # ---------------- ประเมินรายบุคคล ----------------
    with tabs[0]:
        src = st.selectbox("ค่าเริ่มต้นของฟอร์ม",
                           ["ค่าที่พบบ่อยในชุดฝึก"] + [f"ตัวอย่างชุดทดสอบ #{i}" for i in range(10)])
        base = DEFAULTS if src.startswith("ค่า") else X_test.iloc[int(src.split("#")[1])].to_dict()
        with st.form("adv_form"):
            vals = {}
            groups = [("ข้อมูลสินเชื่อ", ["duration", "credit_amount", "purpose",
                                          "installment_commitment", "other_parties",
                                          "other_payment_plans"]),
                      ("สถานะทางการเงิน", ["checking_status", "savings_status", "credit_history",
                                           "existing_credits", "property_magnitude"]),
                      ("ข้อมูลส่วนบุคคล", ["age", "personal_status", "employment", "job",
                                           "housing", "residence_since", "num_dependents",
                                           "own_telephone", "foreign_worker"])]
            listed = [c for _, cs in groups for c in cs]
            groups.append(("อื่น ๆ", [c for c in X_ref.columns if c not in listed]))
            for title, cs in groups:
                cs = [c for c in cs if c in X_ref.columns]
                if not cs:
                    continue
                st.markdown(f"**{title}**")
                cols = st.columns(3)
                for i, c in enumerate(cs):
                    vals[c] = input_widget(c, cols[i % 3], f"a{src}", base[c])
            go_btn = st.form_submit_button("ประเมินความเสี่ยง", type="primary", width="stretch")
        if go_btn:
            show_result(pd.DataFrame([vals])[X_ref.columns], detailed=True)

    # ---------------- ประเมินหลายราย ----------------
    with tabs[1]:
        st.markdown("อัปโหลดไฟล์ CSV ที่มีคอลัมน์ตามชุดข้อมูล (คอลัมน์ที่ขาดจะเติมด้วยค่ากลางของชุดฝึก)")
        st.download_button("ดาวน์โหลดไฟล์ตัวอย่าง", X_test.head(10).to_csv(index=False).encode("utf-8-sig"),
                           "template.csv", "text/csv")
        up = st.file_uploader("เลือกไฟล์ CSV", type="csv")
        if up is not None:
            try:
                batch = pd.read_csv(up)
                missing = [c for c in X_ref.columns if c not in batch.columns]
                for c in missing:
                    batch[c] = DEFAULTS[c]
                if missing:
                    st.warning("เติมค่าให้คอลัมน์ที่ขาด: " + ", ".join(missing))
                Xb = batch[X_ref.columns]
                out = batch.copy()
                for n in MODEL_NAMES:
                    out[f"p_bad_{n}"] = results[n]["pipeline"].predict_proba(Xb)[:, 1]
                out["p_bad"] = out[f"p_bad_{model_name}"]
                out["decision"] = np.where(out["p_bad"] >= thr, "ไม่อนุมัติ", "อนุมัติ")
                k1, k2 = st.columns(2)
                k1.metric("จำนวนรายการ", len(out))
                k2.metric("อัตราอนุมัติ", f"{(out['decision'] == 'อนุมัติ').mean():.1%}")
                st.dataframe(out, width="stretch")
                st.download_button("ดาวน์โหลดผลลัพธ์", out.to_csv(index=False).encode("utf-8-sig"),
                                   "predictions.csv", "text/csv", type="primary")
            except Exception as e:  # noqa: BLE001
                st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")

    # ---------------- EDA ----------------
    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        c1.metric("จำนวนแถว", f"{raw.shape[0]:,}")
        c2.metric("จำนวนฟีเจอร์", raw.shape[1] - 1)
        c3.metric("สัดส่วน 'bad'", f"{(raw['class'] == 'bad').mean():.1%}")
        with st.expander("ตัวอย่างข้อมูล / ค่าที่หายไป / สถิติเชิงพรรณนา"):
            st.dataframe(raw.head(20), width="stretch")
            cc1, cc2 = st.columns([1, 2])
            cc1.dataframe(raw.isna().sum().rename("missing"), width="stretch")
            cc2.dataframe(raw.describe().T, width="stretch")

        num_cols = raw.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in raw.columns if c not in num_cols + ["class"]]
        c1, c2 = st.columns(2)
        with c1:
            num_sel = st.selectbox("ตัวแปรตัวเลข", num_cols, format_func=ft,
                                   index=num_cols.index("credit_amount") if "credit_amount" in num_cols else 0)
            st.plotly_chart(px.box(data, x="class", y=num_sel, color="class",
                                   color_discrete_map={"good": "#4C78A8", "bad": "#E45756"}),
                            width="stretch")
        with c2:
            cat_sel = st.selectbox("ตัวแปรหมวดหมู่", cat_cols + ["sex", "age_group"], format_func=ft)
            rate = (data.groupby(cat_sel)["class"].apply(lambda s: (s == "bad").mean())
                    .rename("bad_rate").reset_index().sort_values("bad_rate"))
            rate["label"] = [vt(cat_sel, v) for v in rate[cat_sel]]
            st.plotly_chart(px.bar(rate, x="bad_rate", y="label", orientation="h", text_auto=".1%",
                                   labels={"bad_rate": "อัตรา bad", "label": ""}), width="stretch")
        st.subheader("Correlation ของตัวแปรตัวเลข")
        st.plotly_chart(px.imshow(raw[num_cols].corr(), text_auto=".2f",
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1), width="stretch")

    # ---------------- โมเดลและ CV ----------------
    with tabs[3]:
        st.markdown(
            f"แบ่ง train/test แบบ stratified (random_state={meta['random_state']}, "
            f"test_size={meta['test_size']}) · GridSearchCV + StratifiedKFold (k={meta['n_splits']}) "
            "บนชุดฝึกเท่านั้น · ตารางใช้ threshold ปัจจุบันของโมเดลที่เลือก")
        st.dataframe(summary_table().style.format(precision=3).highlight_max(
            subset=["CV AUC", "AUC"], color="#d4edda"), width="stretch")
        fig = go.Figure()
        for n, r in results.items():
            fpr, tpr, _ = roc_curve(y_test, r["proba_test"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{n} (AUC {r['test_metrics']['AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                 showlegend=False))
        fig.update_layout(title="ROC curve บนชุดทดสอบ", xaxis_title="FPR", yaxis_title="TPR", height=450)
        st.plotly_chart(fig, width="stretch")
        st.subheader(f"รายละเอียด: {model_name}")
        st.write("Best hyperparameters:", res["best_params"])
        st.dataframe(res["cv_table"], width="stretch")

    # ---------------- Threshold ----------------
    with tabs[4]:
        if "oof_proba" in res and y_train is not None:
            _, curve = tm.find_best_threshold(y_train, res["oof_proba"], cost_fn, cost_fp)
        else:
            curve = res["cost_curve"]
        st.markdown(f"Expected cost = **{cost_fn} × FN + {cost_fp} × FP** จาก out-of-fold predictions บนชุดฝึก")
        fig = px.line(curve, x="threshold", y="cost", title="Expected cost (OOF)")
        fig.add_vline(x=thr, line_dash="dash", annotation_text=f"ใช้อยู่ = {thr:.2f}")
        st.plotly_chart(fig, width="stretch")

        cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Confusion matrix (ชุดทดสอบ)")
            st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                      x=["ทำนาย: good", "ทำนาย: bad"], y=["จริง: good", "จริง: bad"]),
                            width="stretch")
        with c2:
            st.subheader("ตัวชี้วัด")
            st.metric(f"Cost @ {thr:.2f}", cur_metrics["Cost @ best t"],
                      int(cur_metrics["Cost @ best t"] - cur_metrics["Cost @ 0.5"]),
                      delta_color="inverse", help="ส่วนต่างเทียบกับ threshold 0.5")
            st.metric("Cost @ 0.5", cur_metrics["Cost @ 0.5"])
            st.metric("Recall (bad)", f"{cur_metrics['Recall (bad)']:.2f}")
            st.metric("Precision (bad)", f"{cur_metrics['Precision (bad)']:.2f}")

    # ---------------- Importance ----------------
    with tabs[5]:
        imp = res["perm_importance"].copy()
        imp["label"] = imp["feature"].map(ft)
        n_top = st.slider("จำนวนตัวแปรที่แสดง", 5, len(imp), min(15, len(imp)))
        st.plotly_chart(px.bar(imp.head(n_top).iloc[::-1], x="importance_mean", y="label",
                               error_x="importance_std", orientation="h",
                               labels={"importance_mean": "AUC ที่ลดลง", "label": ""},
                               title=f"Permutation importance — {model_name} (ชุดทดสอบ)"),
                        width="stretch")
        st.dataframe(imp.drop(columns="label"), width="stretch")

    # ---------------- Fairness ----------------
    with tabs[6]:
        st.markdown("วัดบนชุดทดสอบด้วย threshold ปัจจุบัน · ผลลัพธ์ที่เป็นประโยชน์ต่อผู้กู้คือ "
                    "**ได้รับอนุมัติ (ทำนายว่า good)**")
        with st.expander("ความหมายของตัวชี้วัด"):
            st.markdown("""
| ตัวชี้วัด | ความหมาย | ค่าที่เป็นธรรม |
|---|---|---|
| Demographic parity diff. | ความต่างของอัตราอนุมัติระหว่างกลุ่ม | ใกล้ 0 |
| Disparate impact ratio | อัตราอนุมัติต่ำสุด / สูงสุด | ≥ 0.8 |
| Equal opportunity diff. (TPR) | ความต่างของอัตราการตรวจพบผู้ที่ bad จริง | ใกล้ 0 |
| FPR diff. | ความต่างของอัตราปฏิเสธผู้ที่ good จริง (= ความต่างของโอกาสได้อนุมัติเมื่อ good จริง) | ใกล้ 0 |
| Equalized odds diff. | ค่ามากสุดของความต่าง TPR และ FPR | ใกล้ 0 |
""")
        for attr in ["sex", "age_group"]:
            st.subheader(f"ตาม {attr}")
            tbl = tm.group_metrics(y_test, pred_test, art["groups_test"][attr])
            st.dataframe(tbl.style.format(precision=3), width="stretch", hide_index=True)
            fs = tm.fairness_summary(tbl)
            for col, (k, v) in zip(st.columns(len(fs)), fs.items()):
                col.metric(k, f"{v:.3f}")
            di = fs["Disparate impact ratio"]
            (st.warning if di < 0.8 else st.success)(
                f"Disparate impact = {di:.2f} ({'ต่ำกว่า' if di < 0.8 else 'ผ่าน'}เกณฑ์ 0.8)")
        if "fairness_compare" in art:
            st.subheader(f"เปรียบเทียบ: ตัด personal_status และ age ออกจากฟีเจอร์ ({best_name}, threshold จาก OOF)")
            st.dataframe(art["fairness_compare"].style.format(precision=3), width="stretch",
                         hide_index=True)
            st.caption("ตัวแปรอื่นอาจยังสะท้อนเพศหรืออายุทางอ้อม (proxy variables)")

st.divider()
st.caption("จัดทำเพื่อการศึกษา · ไม่ใช่คำแนะนำทางการเงิน")
