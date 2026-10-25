"""
app.py  —  Streamlit Application: German Credit Risk
=====================================================
รันในเครื่อง :  streamlit run app.py
Deploy       :  push ขึ้น GitHub แล้วเลือกไฟล์นี้เป็น Main file บน Streamlit Community Cloud

โครงสร้างหน้าจอ (แต่ละแท็บตรงกับข้อในโจทย์)
  แท็บ 0  ภาพรวม           สรุปขั้นตอนและผลลัพธ์หลัก
  แท็บ 1  EDA              ข้อ 1
  แท็บ 2  โมเดล & Threshold ข้อ 2.1–2.4
  แท็บ 3  การอธิบายโมเดล    ข้อ 3
  แท็บ 4  Fairness          ข้อ 4
  แท็บ 5  ทดลองทำนาย       ใช้โมเดลที่เลือกทำนายผู้กู้รายใหม่
งานคำนวณทั้งหมดอยู่ใน credit_core.py ไฟล์นี้ทำหน้าที่แสดงผลเท่านั้น
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

import credit_core as cc

st.set_page_config(page_title="German Credit Risk", page_icon="💳", layout="wide")


# ----------------------------------------------------------------------------
# Cache: Streamlit รันสคริปต์ใหม่ทุกครั้งที่ผู้ใช้กดปุ่มหรือเปลี่ยนค่า
# จึงต้อง cache งานที่ใช้เวลานาน (โหลดข้อมูล, ฝึกโมเดล, SHAP) ไม่ให้คำนวณซ้ำ
#   st.cache_data      -> ผลลัพธ์เป็นข้อมูล (DataFrame)
#   st.cache_resource  -> ผลลัพธ์เป็นออบเจ็กต์ (โมเดลที่ฝึกแล้ว)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="กำลังโหลดข้อมูลจาก OpenML ...")
def get_data():
    return cc.load_data()


@st.cache_resource(show_spinner=False)
def train_all(model_names: tuple, drop_sensitive: bool):
    """ข้อ 2.1–2.4 ทั้งหมด: แยกข้อมูล -> tune -> OOF -> threshold -> ประเมินบน test"""
    df, _ = get_data()
    X, y = cc.make_xy(df, drop_sensitive=drop_sensitive)
    # แยกเฉพาะ X, y (group columns จะดึงตาม index ภายหลัง เพื่อให้เปลี่ยนจุดตัดอายุได้โดยไม่ต้องฝึกใหม่)
    X_train, X_test, y_train, y_test = cc.split_data(X, y)                # ข้อ 2.1
    results = {}
    progress = st.progress(0.0, text="เริ่มฝึกโมเดล ...")
    for i, name in enumerate(model_names):
        progress.progress(i / len(model_names), text=f"กำลัง tune {name} ด้วย StratifiedKFold (k=5) ...")
        search = cc.tune_model(name, X_train, y_train)                     # ข้อ 2.3
        best = search.best_estimator_
        oof = cc.oof_proba(best, X_train, y_train)                          # ข้อ 2.4
        th, oof_cost, curve = cc.find_best_threshold(y_train, oof)
        p_test = best.predict_proba(X_test)[:, 1]
        results[name] = {
            "search": search, "pipeline": best, "oof": oof,
            "threshold": th, "oof_cost": oof_cost, "curve": curve,
            "p_test": p_test,
            "test_at_th": cc.classification_summary(y_test, p_test, th),
            "test_at_05": cc.classification_summary(y_test, p_test, 0.5),
        }
    progress.empty()
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "results": results}


@st.cache_data(show_spinner="กำลังคำนวณ permutation importance ...")
def get_perm_importance(model_name: str, model_names: tuple, drop_sensitive: bool):
    b = train_all(model_names, drop_sensitive)
    return cc.permutation_importance_table(b["results"][model_name]["pipeline"], b["X_test"], b["y_test"])


@st.cache_data(show_spinner="กำลังคำนวณ SHAP ...")
def get_shap(model_name: str, model_names: tuple, drop_sensitive: bool):
    b = train_all(model_names, drop_sensitive)
    return cc.shap_importance_table(b["results"][model_name]["pipeline"], b["X_train"], b["X_test"])


def fmt(x, d=3):
    return f"{x:.{d}f}"


def explain(title: str, body: str):
    """กล่องคำอธิบายขั้นตอน (เปิด/ปิดได้)"""
    with st.expander(f"📘 {title}"):
        st.markdown(body)


# ----------------------------------------------------------------------------
# Sidebar: ตั้งค่า
# ----------------------------------------------------------------------------
df_raw, data_source = get_data()

st.sidebar.header("⚙️ ตั้งค่า")
all_models = list(cc.model_catalog().keys())
chosen = st.sidebar.multiselect("อัลกอริทึมที่ใช้ (อย่างน้อย 4)", all_models, default=all_models)
if len(chosen) < 4:
    st.sidebar.warning("โจทย์กำหนดอย่างน้อย 4 อัลกอริทึม")
if not chosen:
    st.stop()
drop_sensitive = st.sidebar.checkbox(
    "ไม่ใช้ personal_status และ age เป็น feature",
    value=False,
    help="ใช้เปรียบเทียบว่าการตัดตัวแปรอ่อนไหวออกช่วยลดความเหลื่อมล้ำหรือไม่",
)
age_cut = st.sidebar.slider("จุดตัด age_group (ปี)", 20, 40, 25)
st.sidebar.markdown(
    f"**ค่าคงที่ตามโจทย์**  \n"
    f"- positive class = `'{cc.POS_LABEL}'`  \n"
    f"- test size = {cc.TEST_SIZE:.0%}, stratified, random_state = {cc.RANDOM_STATE}  \n"
    f"- StratifiedKFold k = {cc.N_SPLITS}  \n"
    f"- ต้นทุน FN : FP = {cc.COST_FN} : {cc.COST_FP}"
)
st.sidebar.caption(f"แหล่งข้อมูล: {data_source}")

df = cc.add_group_columns(df_raw, age_cut=age_cut)

with st.spinner("กำลังฝึกและปรับจูนโมเดล (ครั้งแรกใช้เวลาประมาณ 1–2 นาที) ..."):
    bundle = train_all(tuple(chosen), drop_sensitive)

X_train, X_test = bundle["X_train"], bundle["X_test"]
y_train, y_test = bundle["y_train"], bundle["y_test"]
R = bundle["results"]
groups_test = df.loc[X_test.index, cc.GROUP_COLS]

# ตารางเปรียบเทียบโมเดล (ใช้หลายแท็บ)
comp = pd.DataFrame([{
    "Model": n,
    "CV ROC-AUC (train)": r["search"].best_score_,
    "Best threshold (OOF)": r["threshold"],
    "OOF expected cost": r["oof_cost"],
    "Test ROC-AUC": r["test_at_th"]["ROC-AUC"],
    "Test expected cost @th": r["test_at_th"]["Expected cost / ราย"],
    "Test expected cost @0.5": r["test_at_05"]["Expected cost / ราย"],
    "Test Recall (bad) @th": r["test_at_th"]["Recall (bad)"],
} for n, r in R.items()]).set_index("Model")

# เลือกโมเดลที่ดีที่สุดจาก OOF expected cost (ข้อมูลชุดฝึก) ไม่ใช้ผลชุดทดสอบตัดสิน เพื่อไม่ให้เกิด leakage
best_name = comp["OOF expected cost"].idxmin()

st.title("💳 German Credit Risk — Model, Explainability & Fairness")
st.caption("Statlog German Credit (OpenML data_id=31) · positive class = 'bad'")

tabs = st.tabs(["0) ภาพรวม", "1) EDA", "2) โมเดล & Threshold",
                "3) การอธิบายโมเดล", "4) Fairness", "5) ทดลองทำนาย"])

# ============================================================================
# แท็บ 0 ภาพรวม
# ============================================================================
with tabs[0]:
    st.subheader("ขั้นตอนการทำงานของแอป")
    st.markdown(
        """
| ข้อ | ขั้นตอน | วิธีที่ใช้ในแอปนี้ |
|---|---|---|
| 1 | โหลดและสำรวจข้อมูล | `fetch_openml(data_id=31)` → ตรวจขนาด ชนิดข้อมูล ค่าว่าง การกระจายตัว และอัตรา bad รายหมวด |
| 2.1 | Target และการแยกข้อมูล | `y = 1` เมื่อ `class == 'bad'` · `train_test_split(stratify=y, random_state=42)` |
| 2.2 | Pipeline | `ColumnTransformer` (StandardScaler + OneHotEncoder) + โมเดล 4 แบบ |
| 2.3 | ปรับ hyperparameter | `GridSearchCV` + `StratifiedKFold(5)` บนชุดฝึกเท่านั้น, scoring = ROC-AUC |
| 2.4 | เลือก threshold | `cross_val_predict` (OOF) → ไล่ threshold 0.01–0.99 → เลือกจุดที่ `(5·FN + 1·FP)/n` ต่ำสุด |
| 3 | อธิบายโมเดล | Permutation importance และ SHAP บนชุดทดสอบ |
| 4 | Fairness | ตัวชี้วัดรายกลุ่มและส่วนต่างระหว่างกลุ่ม ตาม `sex` และ `age_group` |
| 5 | Deploy | GitHub → Streamlit Community Cloud |
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("จำนวนข้อมูล", f"{len(df):,} ราย")
    c2.metric("ชุดฝึก / ชุดทดสอบ", f"{len(X_train)} / {len(X_test)}")
    c3.metric("โมเดลที่ดีที่สุด (OOF cost)", best_name)
    c4.metric("Threshold ที่เลือก", fmt(R[best_name]["threshold"], 2))
    st.dataframe(comp.style.format("{:.3f}").highlight_min(
        subset=["OOF expected cost", "Test expected cost @th"], color="#c8e6c9"),
        width="stretch")
    st.info(
        "การเลือกโมเดลและ threshold ใช้เฉพาะข้อมูลชุดฝึก (CV และ OOF) "
        "ส่วนคอลัมน์ Test ใช้รายงานผลเท่านั้น จึงเป็นการประเมินที่ไม่ลำเอียง"
    )

# ============================================================================
# แท็บ 1 EDA (ข้อ 1)
# ============================================================================
with tabs[1]:
    st.subheader("1) การสำรวจข้อมูล (Exploratory Data Analysis)")
    explain(
        "ขั้นตอนนี้ทำอะไร",
        """
1. โหลดข้อมูลด้วย `fetch_openml(data_id=31, as_frame=True).frame` ได้ 1,000 แถว 21 คอลัมน์
   (ตัวแปรอิสระ 20 ตัว + `class`)
2. ตรวจโครงสร้าง: ชนิดข้อมูล ค่าว่าง และสถิติพื้นฐานของตัวแปรตัวเลข
3. ตรวจความไม่สมดุลของคลาส (good 70% / bad 30%) ซึ่งเป็นเหตุผลที่ต้องแยกข้อมูลแบบ **stratified**
   และไม่ควรใช้ accuracy เป็นตัวชี้วัดหลัก
4. ดูความสัมพันธ์ระหว่างตัวแปรกับอัตรา bad เพื่อทำความเข้าใจข้อมูลก่อนสร้างโมเดล
5. สร้างตัวแปรกลุ่ม `sex` (แยกจาก `personal_status`) และ `age_group` สำหรับวิเคราะห์ fairness ในข้อ 4
""",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("แถว × คอลัมน์", f"{df_raw.shape[0]} × {df_raw.shape[1]}")
    c2.metric("ค่าว่างทั้งหมด", int(df_raw.isna().sum().sum()))
    c3.metric("สัดส่วน bad", f"{(df_raw['class'].astype(str) == 'bad').mean():.0%}")

    st.markdown("**ตัวอย่างข้อมูล**")
    st.dataframe(df_raw.head(10), width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ชนิดข้อมูลและค่าว่าง**")
        info = pd.DataFrame({"dtype": df_raw.dtypes.astype(str),
                             "missing": df_raw.isna().sum(),
                             "unique": df_raw.nunique()})
        st.dataframe(info, width="stretch", height=420)
    with col_b:
        st.markdown("**การกระจายของคลาส**")
        vc = df_raw["class"].astype(str).value_counts().reset_index()
        vc.columns = ["class", "count"]
        st.plotly_chart(px.bar(vc, x="class", y="count", color="class", text="count",
                               color_discrete_map={"good": "#4c9be8", "bad": "#e8704c"}),
                        width="stretch")
        st.markdown("**สถิติพื้นฐานของตัวแปรตัวเลข**")
        st.dataframe(df_raw.describe().T.round(2), width="stretch")

    num_cols = df_raw.select_dtypes("number").columns.tolist()
    cat_cols = [c for c in df_raw.columns if c not in num_cols + ["class"]]

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        v = st.selectbox("ตัวแปรตัวเลข: การกระจายแยกตาม class", num_cols, index=num_cols.index("credit_amount"))
        st.plotly_chart(px.histogram(df_raw.assign(cls=df_raw["class"].astype(str)), x=v, color="cls",
                                     barmode="overlay", nbins=30, marginal="box",
                                     color_discrete_map={"good": "#4c9be8", "bad": "#e8704c"}),
                        width="stretch")
    with col_b:
        c = st.selectbox("ตัวแปรหมวดหมู่: อัตรา bad รายหมวด", cat_cols, index=cat_cols.index("checking_status"))
        t = (df_raw.assign(bad=(df_raw["class"].astype(str) == "bad").astype(int))
             .groupby(c, observed=True)["bad"].agg(["mean", "size"]).reset_index())
        t.columns = [c, "bad_rate", "n"]
        st.plotly_chart(px.bar(t, x=c, y="bad_rate", text="n", hover_data=["n"])
                        .update_layout(yaxis_tickformat=".0%"), width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**สหสัมพันธ์ของตัวแปรตัวเลข (Pearson)**")
        st.plotly_chart(px.imshow(df_raw[num_cols].corr().round(2), text_auto=True,
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                        width="stretch")
    with col_b:
        st.markdown(f"**ตัวแปรกลุ่มสำหรับ fairness (จุดตัดอายุ = {age_cut})**")
        g = (df.assign(bad=(df["class"].astype(str) == "bad").astype(int))
             .melt(id_vars="bad", value_vars=cc.GROUP_COLS, var_name="attribute", value_name="group")
             .groupby(["attribute", "group"])["bad"].agg(n="size", bad_rate="mean").reset_index())
        st.dataframe(g.style.format({"bad_rate": "{:.1%}"}), width="stretch")
        st.caption("ข้อมูลไม่มีคอลัมน์เพศโดยตรง จึงแยกจาก personal_status "
                   "และไม่มีหมวด 'female single' ในข้อมูลชุดนี้")

# ============================================================================
# แท็บ 2 โมเดลและ threshold (ข้อ 2)
# ============================================================================
with tabs[2]:
    st.subheader("2) การสร้างโมเดล การปรับจูน และการเลือก threshold")
    explain(
        "2.1 กำหนด target และแยก train/test",
        f"""
- `y = (class == 'bad').astype(int)` ดังนั้น **1 = bad (positive)**, 0 = good
- `train_test_split(X, y, test_size={cc.TEST_SIZE}, stratify=y, random_state={cc.RANDOM_STATE})`
  ได้ชุดฝึก {len(X_train)} ราย (bad {y_train.mean():.0%}) และชุดทดสอบ {len(X_test)} ราย (bad {y_test.mean():.0%})
- ชุดทดสอบถูกเก็บไว้ใช้ **หลังจาก** เลือกโมเดลและ threshold เสร็จแล้วเท่านั้น
""",
    )
    explain(
        "2.2 Pipeline (ColumnTransformer + โมเดล)",
        """
```python
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
pipe = Pipeline([("prep", prep), ("model", estimator)])
```
อัลกอริทึมที่ใช้: Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF)
การรวม preprocessing ไว้ใน Pipeline ทำให้ทุก fold ของ CV fit scaler/encoder จากข้อมูลฝึกของ fold นั้นเท่านั้น
""",
    )
    explain(
        "2.3 ปรับ hyperparameter ด้วย StratifiedKFold (k=5)",
        """
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv).fit(X_train, y_train)
```
ใช้ ROC-AUC เป็นเกณฑ์เลือก hyperparameter เพราะวัดความสามารถในการจัดอันดับความเสี่ยงโดยไม่ขึ้นกับ threshold
""",
    )
    explain(
        "2.4 เลือก threshold จาก out-of-fold predictions",
        f"""
1. `cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba")`
   ได้ความน่าจะเป็นของแต่ละแถวจากโมเดลที่ไม่เคยเห็นแถวนั้น
2. ไล่ threshold t = 0.01 ถึง 0.99 แล้วคำนวณ
   **Expected cost = ({cc.COST_FN}·FN + {cc.COST_FP}·FP) / n**
3. เลือก t ที่ต้นทุนต่ำสุด แล้วนำไปใช้กับชุดทดสอบ

ค่าอ้างอิงเชิงทฤษฎี: t* = FP/(FP+FN) = {cc.COST_FP}/({cc.COST_FP}+{cc.COST_FN}) ≈ {cc.COST_FP/(cc.COST_FP+cc.COST_FN):.3f}
เมื่อความน่าจะเป็นถูก calibrate ดี threshold ที่ได้จึงควรต่ำกว่า 0.5 มาก
เพราะการปล่อยกู้ให้ลูกหนี้ bad (FN) แพงกว่าการปฏิเสธลูกค้า good (FP) ถึง 5 เท่า
""",
    )

    st.markdown("#### ตารางเปรียบเทียบโมเดล")
    st.dataframe(comp.style.format("{:.3f}").highlight_min(
        subset=["OOF expected cost", "Test expected cost @th"], color="#c8e6c9"),
        width="stretch")
    with st.expander("Hyperparameter ที่ดีที่สุดของแต่ละโมเดล"):
        st.dataframe(pd.DataFrame({n: r["search"].best_params_ for n, r in R.items()}).T.astype(str),
                     width="stretch")

    sel = st.selectbox("เลือกโมเดลเพื่อดูรายละเอียด", list(R), index=list(R).index(best_name))
    r = R[sel]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Expected cost vs threshold (OOF บนชุดฝึก) — {sel}**")
        fig = px.line(r["curve"], x="threshold", y="expected_cost")
        fig.add_vline(x=r["threshold"], line_dash="dash", line_color="green",
                      annotation_text=f"best = {r['threshold']:.2f}")
        fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="0.5")
        fig.add_vline(x=1 / 6, line_dash="dot", line_color="orange", annotation_text="1/6",
                      annotation_position="bottom right")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.markdown("**ROC curve บนชุดทดสอบ (ทุกโมเดล)**")
        fig = go.Figure()
        for n, rr in R.items():
            fpr, tpr, _ = roc_curve(y_test, rr["p_test"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{n} (AUC={rr['test_at_th']['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                 showlegend=False))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", legend=dict(y=0.02, x=0.45))
        st.plotly_chart(fig, width="stretch")

    st.markdown(f"**ผลบนชุดทดสอบ: threshold ที่เลือก ({r['threshold']:.2f}) เทียบกับ 0.5**")
    col_a, col_b, col_c = st.columns([1, 1, 1.2])
    for col, key, label in [(col_a, "test_at_th", f"t = {r['threshold']:.2f}"),
                            (col_b, "test_at_05", "t = 0.50")]:
        s = r[key]
        cm = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]])
        col.plotly_chart(
            px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                      x=["ทาย good", "ทาย bad"], y=["จริง good", "จริง bad"], title=label)
            .update_layout(coloraxis_showscale=False, height=320),
            width="stretch")
    m = pd.DataFrame({f"t={r['threshold']:.2f}": r["test_at_th"], "t=0.50": r["test_at_05"]})
    col_c.dataframe(m.drop(index=["threshold"]).map(lambda v: f"{v:.3f}" if isinstance(v, float) else v),
                    width="stretch", height=420)

# ============================================================================
# แท็บ 3 การอธิบายโมเดล (ข้อ 3)
# ============================================================================
with tabs[3]:
    st.subheader("3) การอธิบายโมเดลบนชุดทดสอบ")
    explain(
        "Permutation importance และ SHAP ต่างกันอย่างไร",
        """
- **Permutation importance**: สลับค่าของตัวแปรหนึ่งในชุดทดสอบแบบสุ่ม (ทำซ้ำ 10 รอบ)
  แล้ววัดว่า ROC-AUC ลดลงเท่าใด ยิ่งลดมาก ตัวแปรนั้นยิ่งสำคัญต่อ "ความแม่นยำ" ของโมเดล
  คำนวณบน Pipeline ทั้งก้อน ผลจึงอยู่ในระดับตัวแปรต้นฉบับ
- **SHAP**: แจกแจงค่าทำนายของแต่ละรายว่าแต่ละตัวแปรดันความน่าจะเป็น bad ขึ้น (+) หรือลง (−) เท่าใด
  ค่าเฉลี่ย |SHAP| ใช้เป็นความสำคัญโดยรวม ส่วนค่ารายแถวใช้อธิบายการตัดสินใจของผู้กู้แต่ละราย
  ค่า SHAP ของคอลัมน์ one-hot ถูกรวมกลับเป็นตัวแปรต้นฉบับ
- ทั้งสองวิธีคำนวณบน **ชุดทดสอบ** ตามโจทย์
""",
    )
    sel3 = st.selectbox("เลือกโมเดล", list(R), index=list(R).index(best_name), key="explain_model")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Permutation importance (ROC-AUC ที่ลดลง)**")
        pi = get_perm_importance(sel3, tuple(chosen), drop_sensitive)
        fig = px.bar(pi.sort_values("importance_mean"), x="importance_mean", y="feature",
                     error_x="importance_std", orientation="h")
        fig.update_layout(height=600, xaxis_title="ROC-AUC ที่ลดลง", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown("**SHAP (ค่าเฉลี่ย |SHAP| บนชุดทดสอบ)**")
        try:
            shap_tab, shap_rows = get_shap(sel3, tuple(chosen), drop_sensitive)
            fig = px.bar(shap_tab.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h")
            fig.update_layout(height=600, xaxis_title="mean |SHAP|", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        except Exception as e:  # noqa: BLE001
            shap_rows = None
            st.warning(f"คำนวณ SHAP ไม่สำเร็จ: {e}")

    if shap_rows is not None:
        st.markdown("**อธิบายรายบุคคล (SHAP ของผู้กู้หนึ่งรายในชุดทดสอบ)**")
        idx = st.selectbox("เลือก index ของผู้กู้", shap_rows.index.tolist())
        p = R[sel3]["pipeline"].predict_proba(X_test.loc[[idx]])[:, 1][0]
        th = R[sel3]["threshold"]
        st.write(f"P(bad) = **{p:.3f}** · threshold = {th:.2f} → ทำนาย **{'bad' if p >= th else 'good'}** "
                 f"· ค่าจริง = **{'bad' if y_test.loc[idx] == 1 else 'good'}**")
        contrib = shap_rows.loc[idx].sort_values(key=np.abs, ascending=False).head(10).iloc[::-1]
        labels = [f"{f} = {X_test.loc[idx, f]}" for f in contrib.index]
        fig = go.Figure(go.Bar(x=contrib.values, y=labels, orientation="h",
                               marker_color=["#e8704c" if v > 0 else "#4c9be8" for v in contrib.values]))
        fig.update_layout(height=420, xaxis_title="SHAP (+ ดันไปทาง bad, − ดันไปทาง good)")
        st.plotly_chart(fig, width="stretch")

# ============================================================================
# แท็บ 4 Fairness (ข้อ 4)
# ============================================================================
with tabs[4]:
    st.subheader("4) Fairness metrics ตาม sex และ age_group (ชุดทดสอบ)")
    explain(
        "นิยามตัวชี้วัด",
        """
| ตัวชี้วัด | ความหมาย | ค่าที่เท่าเทียม |
|---|---|---|
| Selection rate | สัดส่วนที่โมเดลทาย **bad** (ถูกปฏิเสธสินเชื่อ) | เท่ากันทุกกลุ่ม |
| Demographic parity difference | selection rate สูงสุด − ต่ำสุด | 0 |
| Disparate impact ratio | approval rate ต่ำสุด ÷ สูงสุด (approval = ทาย good) | 1 (เกณฑ์ four-fifths ≥ 0.80) |
| Equal opportunity difference | ส่วนต่าง TPR (จับ bad ได้ถูก) | 0 |
| Equalized odds difference | max(ส่วนต่าง TPR, ส่วนต่าง FPR) | 0 |

FPR ในบริบทนี้คือ "ลูกค้าดีที่ถูกปฏิเสธ" จึงเป็นตัวชี้วัดที่สะท้อนความเสียหายต่อผู้กู้โดยตรง
ข้อควรระวัง: ชุดทดสอบมี 200 ราย กลุ่มย่อยบางกลุ่มมีขนาดเล็ก ค่าที่ได้จึงมีความแปรปรวนสูง
""",
    )
    sel4 = st.selectbox("เลือกโมเดล", list(R), index=list(R).index(best_name), key="fair_model")
    th4 = st.slider("Threshold", 0.01, 0.99, float(R[sel4]["threshold"]), 0.01,
                    help="ค่าเริ่มต้นคือ threshold ที่ลด expected cost จาก OOF")
    y_pred = (R[sel4]["p_test"] >= th4).astype(int)

    for attr in cc.GROUP_COLS:
        st.markdown(f"#### ตามตัวแปร `{attr}`")
        tab = cc.fairness_table(y_test, y_pred, groups_test[attr])
        summ = cc.fairness_summary(tab)
        cols = st.columns(4)
        for col, (k, v) in zip(cols, summ.items()):
            flag = ""
            if k.startswith("Disparate"):
                flag = "✅ ผ่าน 4/5" if v >= 0.8 else "⚠️ ต่ำกว่า 0.80"
            col.metric(k, fmt(v), flag, delta_color="off")
        col_a, col_b = st.columns([1.2, 1])
        col_a.dataframe(tab.style.format({c: "{:.3f}" for c in tab.columns if c != "n"}),
                        width="stretch")
        long = tab[["selection_rate (ทาย bad)", "TPR", "FPR"]].reset_index().melt(
            id_vars="group", var_name="metric", value_name="value")
        col_b.plotly_chart(px.bar(long, x="metric", y="value", color="group", barmode="group")
                           .update_layout(height=300, yaxis_tickformat=".0%"),
                           width="stretch")

    st.markdown("#### เปรียบเทียบทุกโมเดล (ที่ threshold ของแต่ละโมเดล)")
    rows = []
    for n, rr in R.items():
        yp = (rr["p_test"] >= rr["threshold"]).astype(int)
        for attr in cc.GROUP_COLS:
            s = cc.fairness_summary(cc.fairness_table(y_test, yp, groups_test[attr]))
            rows.append({"Model": n, "attribute": attr, **s})
    st.dataframe(pd.DataFrame(rows).set_index(["Model", "attribute"]).style.format("{:.3f}"),
                 width="stretch")
    st.caption("ลองติ๊ก 'ไม่ใช้ personal_status และ age เป็น feature' ที่แถบด้านซ้าย "
               "เพื่อเปรียบเทียบผลก่อนและหลังตัดตัวแปรอ่อนไหว (ตัวแปรอื่นอาจยังสัมพันธ์กับอายุหรือเพศอยู่)")

# ============================================================================
# แท็บ 5 ทดลองทำนาย
# ============================================================================
with tabs[5]:
    st.subheader("5) ทดลองทำนายผู้กู้รายใหม่")
    sel5 = st.selectbox("เลือกโมเดล", list(R), index=list(R).index(best_name), key="pred_model")
    feats = X_train.columns.tolist()
    with st.form("predict"):
        cols = st.columns(3)
        values = {}
        for i, f in enumerate(feats):
            col = cols[i % 3]
            if pd.api.types.is_numeric_dtype(X_train[f]):
                values[f] = col.number_input(f, value=float(X_train[f].median()),
                                             min_value=float(X_train[f].min()),
                                             max_value=float(X_train[f].max()))
            else:
                opts = sorted(X_train[f].astype(str).unique())
                mode = X_train[f].astype(str).mode()[0]
                values[f] = col.selectbox(f, opts, index=opts.index(mode))
        submitted = st.form_submit_button("ทำนาย")
    if submitted:
        row = pd.DataFrame([values])
        for f in feats:
            if not pd.api.types.is_numeric_dtype(X_train[f]):
                row[f] = row[f].astype("category")
        p = R[sel5]["pipeline"].predict_proba(row[feats])[:, 1][0]
        th = R[sel5]["threshold"]
        c1, c2, c3 = st.columns(3)
        c1.metric("P(bad)", fmt(p))
        c2.metric("Threshold", fmt(th, 2))
        c3.metric("ผลการพิจารณา", "ปฏิเสธ (bad)" if p >= th else "อนุมัติ (good)")
        st.caption("ผลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่เกณฑ์พิจารณาสินเชื่อจริง")
