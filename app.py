"""
app.py  —  Streamlit Application: German Credit Risk (หน้าเดียว)
================================================================
รันในเครื่อง :  streamlit run app.py
Deploy       :  push ขึ้น GitHub แล้วเลือกไฟล์นี้เป็น Main file บน Streamlit Community Cloud

แนวทางการจัดหน้า
  - ทุกอย่างอยู่ในหน้าเดียว เลื่อนลงตามลำดับข้อสอบ 1 → 5
  - แต่ละหัวข้อแสดง "สรุปผล + กราฟหลัก" ก่อน รายละเอียดเชิงเทคนิคอยู่ในกล่องที่กดเปิดได้
  - แถบด้านซ้ายมีสารบัญ (กดเพื่อเลื่อนไปยังหัวข้อ) และการตั้งค่าทั้งหมด
  - เลือกโมเดลที่ต้องการวิเคราะห์ครั้งเดียวที่แถบด้านซ้าย ใช้ร่วมกันทุกหัวข้อ
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

# สีที่ใช้ทั้งแอป: good = น้ำเงิน, bad = ส้ม (ให้ผู้อ่านจำง่ายตลอดหน้า)
C_GOOD, C_BAD = "#3B82C4", "#E07B39"
CLASS_COLORS = {"good": C_GOOD, "bad": C_BAD}
CHART_H = 340  # ความสูงกราฟมาตรฐาน


# ============================================================================
# ส่วนคำนวณ (มี cache เพื่อไม่ให้ฝึกโมเดลใหม่ทุกครั้งที่ผู้ใช้กดปุ่ม)
#   st.cache_data      -> ผลลัพธ์เป็นข้อมูล (DataFrame)
#   st.cache_resource  -> ผลลัพธ์เป็นออบเจ็กต์ (โมเดลที่ฝึกแล้ว)
# ============================================================================
@st.cache_data(show_spinner="กำลังโหลดข้อมูลจาก OpenML ...")
def get_data():
    return cc.load_data()


@st.cache_resource(show_spinner=False)
def train_all(model_names: tuple, drop_sensitive: bool):
    """ข้อ 2.1–2.4: แยกข้อมูล -> tune -> OOF -> threshold -> ประเมินบน test"""
    df, _ = get_data()
    X, y = cc.make_xy(df, drop_sensitive=drop_sensitive)
    X_train, X_test, y_train, y_test = cc.split_data(X, y)                # ข้อ 2.1
    results = {}
    bar = st.progress(0.0, text="เริ่มฝึกโมเดล ...")
    for i, name in enumerate(model_names):
        bar.progress(i / len(model_names), text=f"กำลังปรับจูน {name} ({i + 1}/{len(model_names)}) ...")
        search = cc.tune_model(name, X_train, y_train)                     # ข้อ 2.3
        best = search.best_estimator_
        oof = cc.oof_proba(best, X_train, y_train)                          # ข้อ 2.4
        th, oof_cost, curve = cc.find_best_threshold(y_train, oof)
        p_test = best.predict_proba(X_test)[:, 1]
        results[name] = {
            "search": search, "pipeline": best,
            "threshold": th, "oof_cost": oof_cost, "curve": curve,
            "p_test": p_test,
            "test_at_th": cc.classification_summary(y_test, p_test, th),
            "test_at_05": cc.classification_summary(y_test, p_test, 0.5),
        }
    bar.empty()
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "results": results}


@st.cache_data(show_spinner="กำลังคำนวณ permutation importance ...")
def get_perm_importance(model_name, model_names, drop_sensitive):
    b = train_all(model_names, drop_sensitive)
    return cc.permutation_importance_table(b["results"][model_name]["pipeline"], b["X_test"], b["y_test"])


@st.cache_data(show_spinner="กำลังคำนวณ SHAP ...")
def get_shap(model_name, model_names, drop_sensitive):
    b = train_all(model_names, drop_sensitive)
    return cc.shap_importance_table(b["results"][model_name]["pipeline"], b["X_train"], b["X_test"])


# ============================================================================
# ฟังก์ชันช่วยจัดหน้า
# ============================================================================
def section(num: str, title: str, anchor: str, subtitle: str = ""):
    """หัวข้อใหญ่ของแต่ละส่วน + เส้นคั่น (anchor ใช้กับลิงก์สารบัญ)"""
    st.divider()
    st.header(f"{num}  {title}", anchor=anchor)
    if subtitle:
        st.caption(subtitle)


def how_it_works(body: str):
    """กล่อง 'วิธีการ' ที่กดเปิดได้ เก็บรายละเอียดเชิงเทคนิคไว้ไม่ให้รกหน้า"""
    with st.expander("📘 วิธีการและโค้ดที่ใช้ในขั้นตอนนี้"):
        st.markdown(body)


def bar_h(df_, x, y, xlabel, color=C_GOOD, error=None, height=CHART_H):
    """กราฟแท่งแนวนอนแบบมาตรฐาน (เรียงค่ามากไว้บนสุด)"""
    fig = px.bar(df_.iloc[::-1], x=x, y=y, orientation="h", error_x=error,
                 color_discrete_sequence=[color])
    fig.update_layout(height=height, xaxis_title=xlabel, yaxis_title="", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def tidy(fig, height=CHART_H, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), **kw)
    return fig


# ============================================================================
# แถบด้านซ้าย: สารบัญ + การตั้งค่า
# ============================================================================
df_raw, data_source = get_data()
all_models = list(cc.model_catalog().keys())

with st.sidebar:
    st.markdown("### 📑 สารบัญ")
    st.markdown(
        "- [สรุปผลภาพรวม](#summary)\n"
        "- [1. สำรวจข้อมูล](#eda)\n"
        "- [2. โมเดลและ threshold](#model)\n"
        "- [3. การอธิบายโมเดล](#explain)\n"
        "- [4. ความเป็นธรรม](#fairness)\n"
        "- [5. ทดลองทำนาย](#predict)"
    )
    st.divider()
    st.markdown("### ⚙️ การตั้งค่า")
    chosen = st.multiselect("อัลกอริทึม (โจทย์กำหนด ≥ 4)", all_models, default=all_models)
    if len(chosen) < 4:
        st.warning("โจทย์กำหนดอย่างน้อย 4 อัลกอริทึม")
    if not chosen:
        st.stop()
    drop_sensitive = st.checkbox(
        "ไม่ใช้ personal_status และ age เป็น feature",
        help="เปรียบเทียบว่าการตัดตัวแปรอ่อนไหวออกช่วยลดความเหลื่อมล้ำหรือไม่",
    )
    age_cut = st.slider("จุดตัด age_group (ปี)", 20, 40, 25)

df = cc.add_group_columns(df_raw, age_cut=age_cut)

with st.spinner("กำลังฝึกและปรับจูนโมเดล (ครั้งแรกประมาณ 1–2 นาที) ..."):
    bundle = train_all(tuple(chosen), drop_sensitive)

X_train, X_test = bundle["X_train"], bundle["X_test"]
y_train, y_test = bundle["y_train"], bundle["y_test"]
R = bundle["results"]
groups_test = df.loc[X_test.index, cc.GROUP_COLS]

# โมเดลที่ดีที่สุด = OOF expected cost ต่ำสุด (ใช้ข้อมูลชุดฝึกเท่านั้น ไม่ใช้ชุดทดสอบตัดสิน)
best_name = min(R, key=lambda n: R[n]["oof_cost"])

with st.sidebar:
    focus = st.selectbox("โมเดลที่ใช้วิเคราะห์ในหัวข้อ 2–5", list(R), index=list(R).index(best_name),
                         help="ค่าเริ่มต้นคือโมเดลที่มี expected cost (OOF) ต่ำสุด")
    st.divider()
    st.caption(
        f"positive class = '{cc.POS_LABEL}' · test {cc.TEST_SIZE:.0%} stratified · "
        f"random_state {cc.RANDOM_STATE} · k = {cc.N_SPLITS} · FN:FP = {cc.COST_FN}:{cc.COST_FP}"
    )
    st.caption(f"แหล่งข้อมูล: {data_source}")

F = R[focus]                      # ผลของโมเดลที่เลือกวิเคราะห์
S = F["test_at_th"]               # ผลบนชุดทดสอบที่ threshold ที่เลือก

# ============================================================================
# ส่วนหัว + สรุปผลภาพรวม
# ============================================================================
st.title("💳 German Credit Risk")
st.markdown(
    "วิเคราะห์ความเสี่ยงสินเชื่อจากชุดข้อมูล **Statlog German Credit** (OpenML data_id=31) "
    "ตั้งแต่สำรวจข้อมูล สร้างโมเดล เลือกเกณฑ์ตัดสินตามต้นทุน อธิบายโมเดล ไปจนถึงตรวจความเป็นธรรม"
)

st.subheader("สรุปผลภาพรวม", anchor="summary")
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("โมเดลที่ดีที่สุด", best_name, help="เลือกจาก expected cost ต่ำสุดบนชุดฝึก (OOF)")
    c2.metric("Threshold ที่เลือก", f"{R[best_name]['threshold']:.2f}", help="ทำนาย bad เมื่อ P(bad) ≥ ค่านี้")
    c3.metric("ROC-AUC (ชุดทดสอบ)", f"{R[best_name]['test_at_th']['ROC-AUC']:.3f}")
    c4.metric("จับลูกหนี้ bad ได้ (Recall)", f"{R[best_name]['test_at_th']['Recall (bad)']:.0%}")
    saved = R[best_name]["test_at_05"]["Total cost"] - R[best_name]["test_at_th"]["Total cost"]
    st.markdown(
        f"เมื่อใช้ threshold **{R[best_name]['threshold']:.2f}** แทน 0.50 "
        f"ต้นทุนรวมบนชุดทดสอบ {'ลดลง' if saved >= 0 else 'เพิ่มขึ้น'} **{abs(saved)} หน่วย** "
        f"({R[best_name]['test_at_05']['Total cost']} → {R[best_name]['test_at_th']['Total cost']}) "
        f"เพราะการปล่อยกู้ให้ลูกหนี้ bad มีต้นทุนสูงกว่าการปฏิเสธลูกค้า good ถึง {cc.COST_FN} เท่า"
    )

# ============================================================================
# 1. สำรวจข้อมูล (EDA)
# ============================================================================
section("1.", "สำรวจข้อมูล (EDA)", "eda", "ข้อ 1 · โหลดข้อมูลด้วย fetch_openml(data_id=31)")

bad_rate = (df_raw["class"].astype(str) == "bad").mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนผู้กู้", f"{len(df_raw):,} ราย")
c2.metric("ตัวแปรอิสระ", f"{df_raw.shape[1] - 1} ตัว")
c3.metric("ค่าว่าง", int(df_raw.isna().sum().sum()))
c4.metric("สัดส่วน bad", f"{bad_rate:.0%}")

num_cols = df_raw.select_dtypes("number").columns.tolist()
cat_cols = [c for c in df_raw.columns if c not in num_cols + ["class"]]
eda = df_raw.assign(cls=df_raw["class"].astype(str), bad=(df_raw["class"].astype(str) == "bad").astype(int))

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**การกระจายของคลาส**")
    vc = eda["cls"].value_counts().rename_axis("class").reset_index(name="count")
    st.plotly_chart(tidy(px.bar(vc, x="class", y="count", color="class", text="count",
                                color_discrete_map=CLASS_COLORS), showlegend=False), width="stretch")
with col_b:
    c = st.selectbox("**อัตรา bad รายหมวด** — เลือกตัวแปร", cat_cols, index=cat_cols.index("checking_status"))
    t = eda.groupby(c, observed=True)["bad"].agg(bad_rate="mean", n="size").reset_index()
    fig = px.bar(t, x=c, y="bad_rate", text=t["bad_rate"].map("{:.0%}".format),
                 color_discrete_sequence=[C_BAD], hover_data=["n"])
    fig.add_hline(y=bad_rate, line_dash="dot", annotation_text=f"เฉลี่ย {bad_rate:.0%}")
    st.plotly_chart(tidy(fig, yaxis_tickformat=".0%", xaxis_title=""), width="stretch")

st.info(
    f"ข้อมูลไม่สมดุล (good {1 - bad_rate:.0%} / bad {bad_rate:.0%}) จึงต้องแยกข้อมูลแบบ stratified "
    "และไม่ใช้ accuracy เป็นตัวชี้วัดหลัก · ไม่มีค่าว่าง จึงไม่ต้องเติมข้อมูล"
)

with st.expander("🔍 ดูข้อมูลและสถิติเพิ่มเติม"):
    st.markdown("**ตัวอย่างข้อมูล 10 แถวแรก**")
    st.dataframe(df_raw.head(10), width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**ชนิดข้อมูลและจำนวนค่าที่ไม่ซ้ำ**")
        st.dataframe(pd.DataFrame({"dtype": df_raw.dtypes.astype(str), "missing": df_raw.isna().sum(),
                                   "unique": df_raw.nunique()}), width="stretch", height=300)
    with col_b:
        st.markdown("**สถิติพื้นฐานของตัวแปรตัวเลข**")
        st.dataframe(df_raw.describe().T.round(1), width="stretch", height=300)
    col_a, col_b = st.columns(2)
    with col_a:
        v = st.selectbox("การกระจายของตัวแปรตัวเลขแยกตาม class", num_cols, index=num_cols.index("credit_amount"))
        st.plotly_chart(tidy(px.histogram(eda, x=v, color="cls", barmode="overlay", nbins=30,
                                          color_discrete_map=CLASS_COLORS), legend_title=""),
                        width="stretch")
    with col_b:
        st.markdown("**สหสัมพันธ์ของตัวแปรตัวเลข**")
        st.plotly_chart(tidy(px.imshow(df_raw[num_cols].corr().round(2), text_auto=True,
                                       color_continuous_scale="RdBu_r", zmin=-1, zmax=1)),
                        width="stretch")
    st.markdown(f"**ตัวแปรกลุ่มสำหรับวิเคราะห์ความเป็นธรรม** (จุดตัดอายุ = {age_cut} ปี)")
    g = (df.assign(bad=(df["class"].astype(str) == "bad").astype(int))
         .melt(id_vars="bad", value_vars=cc.GROUP_COLS, var_name="ตัวแปร", value_name="กลุ่ม")
         .groupby(["ตัวแปร", "กลุ่ม"])["bad"].agg(จำนวน="size", อัตรา_bad="mean").reset_index())
    st.dataframe(g.style.format({"อัตรา_bad": "{:.1%}"}), width="stretch", hide_index=True)
    st.caption("ข้อมูลไม่มีคอลัมน์เพศโดยตรง จึงแยก sex จาก personal_status · "
               "ตัวแปร sex และ age_group ใช้จัดกลุ่มเท่านั้น ไม่ได้ป้อนเข้าโมเดล")

how_it_works(
    """
1. `df = fetch_openml(data_id=31, as_frame=True).frame` → 1,000 แถว 21 คอลัมน์ (ถ้าโหลดไม่ได้จะใช้ไฟล์สำรองใน `data/`)
2. ตรวจชนิดข้อมูล ค่าว่าง สถิติพื้นฐาน และสัดส่วนคลาส
3. ดูอัตรา bad รายหมวด การกระจายตัวแปรตัวเลข และสหสัมพันธ์
4. สร้าง `sex` จาก `personal_status` และ `age_group` จาก `age` สำหรับข้อ 4
"""
)

# ============================================================================
# 2. โมเดลและ threshold
# ============================================================================
section("2.", "สร้างโมเดลและเลือก threshold", "model",
        "ข้อ 2.1–2.4 · Pipeline 4 อัลกอริทึม · StratifiedKFold (k=5) · ต้นทุน FN:FP = 5:1")

st.markdown("**เปรียบเทียบโมเดล** (แถวที่ไฮไลต์ = expected cost ต่ำสุด)")
comp = pd.DataFrame([{
    "โมเดล": n,
    "ROC-AUC (CV)": r["search"].best_score_,
    "Threshold": r["threshold"],
    "Cost/ราย (OOF)": r["oof_cost"],
    "ROC-AUC (test)": r["test_at_th"]["ROC-AUC"],
    "Cost/ราย (test)": r["test_at_th"]["Expected cost / ราย"],
    "Recall bad (test)": r["test_at_th"]["Recall (bad)"],
} for n, r in R.items()]).set_index("โมเดล")
st.dataframe(
    comp.style.format("{:.3f}").format({"Threshold": "{:.2f}", "Recall bad (test)": "{:.0%}"})
    .highlight_min(subset=["Cost/ราย (OOF)"], axis=0, props="background-color:#d9f2dd; font-weight:600"),
    width="stretch",
)
st.caption("OOF = ค่าจากชุดฝึกแบบ out-of-fold ใช้เลือกโมเดลและ threshold · test = ชุดทดสอบ ใช้รายงานผลเท่านั้น")

st.markdown(f"**รายละเอียดของ {focus}**")
col_a, col_b = st.columns(2)
with col_a:
    fig = px.line(F["curve"], x="threshold", y="expected_cost", color_discrete_sequence=[C_GOOD])
    fig.add_vline(x=F["threshold"], line_dash="dash", line_color="green",
                  annotation_text=f"เลือก {F['threshold']:.2f}")
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="0.50")
    st.plotly_chart(tidy(fig, title="Expected cost ตาม threshold (OOF)", yaxis_title="cost / ราย"),
                    width="stretch")
with col_b:
    cm = np.array([[S["TN"], S["FP"]], [S["FN"], S["TP"]]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["ทาย good", "ทาย bad"], y=["จริง good", "จริง bad"])
    st.plotly_chart(tidy(fig, title=f"Confusion matrix บนชุดทดสอบ (t = {F['threshold']:.2f})",
                         coloraxis_showscale=False), width="stretch")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{S['Accuracy']:.1%}")
c2.metric("Precision (bad)", f"{S['Precision (bad)']:.1%}")
c3.metric("Recall (bad)", f"{S['Recall (bad)']:.1%}")
c4.metric("ต้นทุนรวม (test)", f"{S['Total cost']}",
          f"{S['Total cost'] - F['test_at_05']['Total cost']} เทียบกับ t=0.50", delta_color="inverse")

with st.expander("📈 ROC curve ทุกโมเดล และผลเทียบ threshold 0.50"):
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        for n, rr in R.items():
            fpr, tpr, _ = roc_curve(y_test, rr["p_test"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{n} ({rr['test_at_th']['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
                                 showlegend=False))
        st.plotly_chart(tidy(fig, height=380, title="ROC curve (ชุดทดสอบ)", xaxis_title="FPR",
                             yaxis_title="TPR", legend=dict(x=0.4, y=0.05)), width="stretch")
    with col_b:
        keys = ["Accuracy", "Precision (bad)", "Recall (bad)", "F1 (bad)", "TN", "FP", "FN", "TP", "Total cost"]
        m = pd.DataFrame({f"t = {F['threshold']:.2f}": {k: S[k] for k in keys},
                          "t = 0.50": {k: F["test_at_05"][k] for k in keys}})
        st.markdown(f"**{focus}: threshold ที่เลือก เทียบกับ 0.50**")
        st.dataframe(m.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v), width="stretch")
    st.markdown("**Hyperparameter ที่ดีที่สุดของแต่ละโมเดล**")
    st.dataframe(pd.DataFrame({n: r["search"].best_params_ for n, r in R.items()}).T.astype(str), width="stretch")

how_it_works(
    f"""
**2.1 Target และการแยกข้อมูล** — `y = (class == 'bad')` ดังนั้น 1 = bad ·
`train_test_split(test_size=0.2, stratify=y, random_state=42)` ได้ชุดฝึก {len(X_train)} ราย / ทดสอบ {len(X_test)} ราย

**2.2 Pipeline**
```python
prep = ColumnTransformer([("num", StandardScaler(), num_cols),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
pipe = Pipeline([("prep", prep), ("model", estimator)])   # LR, RF, GB, SVM
```

**2.3 ปรับ hyperparameter** — `GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=42))`
ใช้เฉพาะ `X_train, y_train`

**2.4 เลือก threshold** — `cross_val_predict(..., method="predict_proba")` ได้ค่า OOF →
ไล่ t = 0.01–0.99 → เลือก t ที่ `(5·FN + 1·FP) / n` ต่ำสุด
(ค่าอ้างอิงเชิงทฤษฎี 1/(1+5) ≈ 0.167)
"""
)

# ============================================================================
# 3. การอธิบายโมเดล
# ============================================================================
section("3.", "การอธิบายโมเดล", "explain", f"ข้อ 3 · Permutation importance และ SHAP บนชุดทดสอบ · {focus}")

pi = get_perm_importance(focus, tuple(chosen), drop_sensitive)
try:
    shap_tab, shap_rows = get_shap(focus, tuple(chosen), drop_sensitive)
except Exception as e:  # noqa: BLE001
    shap_tab = shap_rows = None
    st.warning(f"คำนวณ SHAP ไม่สำเร็จ: {e}")

TOP = 10
col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Permutation importance — {TOP} อันดับแรก**")
    st.plotly_chart(bar_h(pi.head(TOP), "importance_mean", "feature", "ROC-AUC ที่ลดลงเมื่อสลับค่า",
                          error="importance_std"), width="stretch")
with col_b:
    if shap_tab is not None:
        st.markdown(f"**SHAP (ค่าเฉลี่ย |SHAP|) — {TOP} อันดับแรก**")
        st.plotly_chart(bar_h(shap_tab.head(TOP), "mean_abs_shap", "feature", "ผลต่อ P(bad) โดยเฉลี่ย",
                              color=C_BAD), width="stretch")

top_pi = pi["feature"].head(3).tolist()
st.info(f"ตัวแปรที่มีผลมากที่สุดตาม permutation importance: **{', '.join(top_pi)}**")

if shap_rows is not None:
    with st.expander("👤 อธิบายผลทำนายรายบุคคล (SHAP)"):
        idx = st.selectbox("เลือกผู้กู้ในชุดทดสอบ (index)", shap_rows.index.tolist())
        p = F["pipeline"].predict_proba(X_test.loc[[idx]])[:, 1][0]
        c1, c2, c3 = st.columns(3)
        c1.metric("P(bad)", f"{p:.3f}")
        c2.metric("ผลทำนาย", "bad" if p >= F["threshold"] else "good")
        c3.metric("ค่าจริง", "bad" if y_test.loc[idx] == 1 else "good")
        contrib = shap_rows.loc[idx].sort_values(key=np.abs, ascending=False).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=contrib.values, y=[f"{f} = {X_test.loc[idx, f]}" for f in contrib.index],
                               orientation="h",
                               marker_color=[C_BAD if v > 0 else C_GOOD for v in contrib.values]))
        st.plotly_chart(tidy(fig, height=380, xaxis_title="สีส้ม = ดันไปทาง bad · สีน้ำเงิน = ดันไปทาง good"),
                        width="stretch")

how_it_works(
    """
- **Permutation importance**: สลับค่าตัวแปรทีละตัวในชุดทดสอบ (10 รอบ) แล้ววัดว่า ROC-AUC ลดลงเท่าใด
  คำนวณบน Pipeline ทั้งก้อน ผลจึงเป็นระดับตัวแปรต้นฉบับ
- **SHAP**: แจกแจงว่าแต่ละตัวแปรดัน P(bad) ขึ้นหรือลงเท่าใดในแต่ละราย
  ใช้ TreeExplainer กับ RF/GB และ Permutation explainer กับ LR/SVM แล้วรวมค่าของคอลัมน์ one-hot กลับเป็นตัวแปรเดิม
"""
)

# ============================================================================
# 4. ความเป็นธรรม
# ============================================================================
section("4.", "ความเป็นธรรม (Fairness)", "fairness", f"ข้อ 4 · ตาม sex และ age_group บนชุดทดสอบ · {focus}")

th4 = st.slider("Threshold ที่ใช้ตรวจ (ลองเลื่อนเพื่อดูผลกระทบ)", 0.01, 0.99, float(F["threshold"]), 0.01)
y_pred = (F["p_test"] >= th4).astype(int)

cols = st.columns(2)
fair_tables = {}
for col, attr in zip(cols, cc.GROUP_COLS):
    tab = cc.fairness_table(y_test, y_pred, groups_test[attr])
    summ = cc.fairness_summary(tab)
    fair_tables[attr] = (tab, summ)
    di = summ["Disparate impact ratio (approval)"]
    with col, st.container(border=True):
        st.markdown(f"#### ตาม `{attr}`")
        m1, m2 = st.columns(2)
        m1.metric("Disparate impact", f"{di:.2f}", "ผ่านเกณฑ์ 0.80" if di >= 0.8 else "ต่ำกว่า 0.80",
                  delta_color="normal" if di >= 0.8 else "inverse")
        m2.metric("Equalized odds diff.", f"{summ['Equalized odds difference']:.3f}")
        long = (tab[["selection_rate (ทาย bad)", "TPR", "FPR"]]
                .rename(columns={"selection_rate (ทาย bad)": "ถูกทาย bad"})
                .reset_index().melt(id_vars="group", var_name="ตัวชี้วัด", value_name="ค่า"))
        st.plotly_chart(tidy(px.bar(long, x="ตัวชี้วัด", y="ค่า", color="group", barmode="group", text_auto=".0%"),
                             height=300, yaxis_tickformat=".0%", legend_title="", xaxis_title=""),
                        width="stretch")

st.caption("Disparate impact = อัตราอนุมัติของกลุ่มต่ำสุด ÷ สูงสุด (ควร ≥ 0.80) · "
           "Equalized odds difference = ส่วนต่าง TPR/FPR ที่มากกว่า (ยิ่งใกล้ 0 ยิ่งเท่าเทียม)")

with st.expander("📋 ตารางตัวชี้วัดรายกลุ่ม และเปรียบเทียบทุกโมเดล"):
    for attr, (tab, summ) in fair_tables.items():
        st.markdown(f"**{attr}**")
        st.dataframe(tab.style.format({c: "{:.3f}" for c in tab.columns if c != "n"}), width="stretch")
        st.dataframe(pd.DataFrame([summ]).style.format("{:.3f}"), width="stretch", hide_index=True)
    st.markdown("**ทุกโมเดล (ที่ threshold ของแต่ละโมเดล)**")
    rows = []
    for n, rr in R.items():
        yp = (rr["p_test"] >= rr["threshold"]).astype(int)
        for attr in cc.GROUP_COLS:
            rows.append({"Model": n, "attribute": attr,
                         **cc.fairness_summary(cc.fairness_table(y_test, yp, groups_test[attr]))})
    st.dataframe(pd.DataFrame(rows).set_index(["Model", "attribute"]).style.format("{:.3f}"), width="stretch")

how_it_works(
    """
| ตัวชี้วัด | ความหมาย | ค่าที่เท่าเทียม |
|---|---|---|
| Selection rate | สัดส่วนที่ถูกทาย bad (ถูกปฏิเสธ) | เท่ากันทุกกลุ่ม |
| Demographic parity difference | selection rate สูงสุด − ต่ำสุด | 0 |
| Disparate impact ratio | approval rate ต่ำสุด ÷ สูงสุด | ≥ 0.80 |
| Equal opportunity difference | ส่วนต่าง TPR | 0 |
| Equalized odds difference | max(ส่วนต่าง TPR, ส่วนต่าง FPR) | 0 |

ข้อควรระวัง: ชุดทดสอบมี 200 ราย กลุ่มย่อยบางกลุ่มมีขนาดเล็ก ค่าจึงมีความแปรปรวนสูง ·
ลองติ๊ก "ไม่ใช้ personal_status และ age เป็น feature" ที่แถบด้านซ้ายเพื่อเปรียบเทียบผล
"""
)

# ============================================================================
# 5. ทดลองทำนาย
# ============================================================================
section("5.", "ทดลองทำนายผู้กู้รายใหม่", "predict", f"ใช้โมเดล {focus} และ threshold {F['threshold']:.2f}")

feats = X_train.columns.tolist()
with st.form("predict"):
    cols = st.columns(4)
    values = {}
    for i, f in enumerate(feats):
        col = cols[i % 4]
        if pd.api.types.is_numeric_dtype(X_train[f]):
            values[f] = col.number_input(f, value=float(X_train[f].median()),
                                         min_value=float(X_train[f].min()), max_value=float(X_train[f].max()))
        else:
            opts = sorted(X_train[f].astype(str).unique())
            values[f] = col.selectbox(f, opts, index=opts.index(X_train[f].astype(str).mode()[0]))
    submitted = st.form_submit_button("ทำนายผล", type="primary")

if submitted:
    row = pd.DataFrame([values])
    for f in feats:
        if not pd.api.types.is_numeric_dtype(X_train[f]):
            row[f] = row[f].astype("category")
    p = F["pipeline"].predict_proba(row[feats])[:, 1][0]
    if p >= F["threshold"]:
        st.error(f"ผลการพิจารณา: **ปฏิเสธ (bad)** · P(bad) = {p:.3f} ≥ threshold {F['threshold']:.2f}")
    else:
        st.success(f"ผลการพิจารณา: **อนุมัติ (good)** · P(bad) = {p:.3f} < threshold {F['threshold']:.2f}")
    st.caption("ผลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่เกณฑ์พิจารณาสินเชื่อจริง")
