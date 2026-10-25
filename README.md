# German Credit Risk — Streamlit Application

แอปพลิเคชัน Streamlit สำหรับวิเคราะห์ความเสี่ยงสินเชื่อจากชุดข้อมูล **Statlog German Credit (OpenML data_id=31)**
ครอบคลุมการสำรวจข้อมูล การสร้างและปรับจูนโมเดล การเลือก threshold ตามต้นทุน การอธิบายโมเดล และการวัดความเป็นธรรม

---

## 1. โครงสร้างไฟล์

```
german-credit-streamlit/
├── app.py                    # หน้าจอ Streamlit (แท็บตามข้อสอบ)
├── credit_core.py            # ฟังก์ชันคำนวณทุกขั้นตอน (มีคำอธิบายในโค้ด)
├── requirements.txt          # ไลบรารีที่ Streamlit Cloud ต้องติดตั้ง
├── data/
│   └── credit_g_backup.csv   # ไฟล์สำรอง ใช้เมื่อโหลดจาก OpenML ไม่ได้
├── .streamlit/config.toml    # ตั้งค่าธีมและเซิร์ฟเวอร์
├── .gitignore
└── README.md
```

แยกงานเป็น 2 ไฟล์โดยตั้งใจ: `credit_core.py` เก็บตรรกะการคำนวณ ส่วน `app.py` ทำหน้าที่แสดงผล
ทำให้ตรวจโค้ดแต่ละขั้นได้ง่าย และทดสอบการคำนวณได้โดยไม่ต้องเปิดหน้าเว็บ

## 2. ขั้นตอนการพัฒนาโค้ด (ตรงกับข้อสอบ)

### ข้อ 1 โหลดข้อมูลและ EDA — `load_data()`, `add_group_columns()`, แท็บ "1) EDA"
1. โหลดด้วยคำสั่งตามโจทย์ `fetch_openml(data_id=31, as_frame=True).frame` (1,000 แถว, 21 คอลัมน์)
2. ตรวจชนิดข้อมูล ค่าว่าง สถิติพื้นฐาน และสัดส่วนคลาส (good 700 / bad 300)
3. ดูการกระจายของตัวแปรตัวเลขแยกตามคลาส อัตรา bad รายหมวดของตัวแปรหมวดหมู่ และสหสัมพันธ์
4. สร้างตัวแปรกลุ่มสำหรับ fairness
   - `sex` แยกจาก `personal_status` (ข้อมูลไม่มีคอลัมน์เพศโดยตรง)
   - `age_group` แบ่งที่อายุ 25 ปี (ปรับได้ที่แถบด้านซ้าย)
   - ทั้งสองคอลัมน์ใช้จัดกลุ่มเท่านั้น ไม่ได้ป้อนเข้าโมเดล
5. หากเชื่อมต่อ OpenML ไม่ได้ จะอ่าน `data/credit_g_backup.csv` แทน
   (แปลงจากไฟล์ UCI ต้นฉบับให้ชื่อคอลัมน์และค่าหมวดหมู่ตรงกับ OpenML) แหล่งที่ใช้จริงแสดงที่แถบด้านซ้าย

### ข้อ 2.1 Target และการแยกข้อมูล — `make_xy()`, `split_data()`
```python
y = (df["class"] == "bad").astype(int)          # positive class = 'bad'
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

### ข้อ 2.2 Pipeline — `build_preprocessor()`, `model_catalog()`
```python
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
pipe = Pipeline([("prep", prep), ("model", estimator)])
```
อัลกอริทึม 4 แบบ: Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF)

### ข้อ 2.3 ปรับ hyperparameter — `tune_model()`
`GridSearchCV` + `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ใช้ ROC-AUC เป็นเกณฑ์
และส่งเฉพาะ `X_train, y_train` เข้าไป ชุดทดสอบไม่ถูกใช้ในขั้นนี้

### ข้อ 2.4 เลือก threshold — `oof_proba()`, `find_best_threshold()`
1. `cross_val_predict(..., method="predict_proba")` ได้ค่าทำนายแบบ out-of-fold ของชุดฝึก
2. ไล่ threshold 0.01–0.99 คำนวณ `Expected cost = (5·FN + 1·FP) / n`
3. เลือก threshold ที่ต้นทุนต่ำสุด และเลือก "โมเดลที่ดีที่สุด" จาก OOF cost (ไม่ใช้ผลชุดทดสอบตัดสิน)
4. ค่าอ้างอิงเชิงทฤษฎีคือ 1/(1+5) ≈ 0.167

### ข้อ 3 การอธิบายโมเดล — `permutation_importance_table()`, `shap_importance_table()`
- Permutation importance บนชุดทดสอบ (ROC-AUC ที่ลดลง, ทำซ้ำ 10 รอบ)
- SHAP บนชุดทดสอบ: TreeExplainer สำหรับ RF/GB, Permutation explainer สำหรับ LR/SVM
  รวมค่า SHAP ของคอลัมน์ one-hot กลับเป็นตัวแปรต้นฉบับ และมีกราฟอธิบายรายบุคคล

### ข้อ 4 Fairness — `fairness_table()`, `fairness_summary()`
รายกลุ่ม: selection rate, approval rate, base rate, TPR, FPR, FNR, accuracy
ระหว่างกลุ่ม: demographic parity difference, disparate impact ratio (เกณฑ์ 0.80),
equal opportunity difference, equalized odds difference — คำนวณแยกตาม `sex` และ `age_group`

### ข้อ 5 Application — `app.py`
| แท็บ | เนื้อหา |
|---|---|
| 0) ภาพรวม | สรุปขั้นตอนและตารางเปรียบเทียบโมเดล |
| 1) EDA | ข้อ 1 |
| 2) โมเดล & Threshold | ข้อ 2.1–2.4, กราฟ cost–threshold, ROC, confusion matrix |
| 3) การอธิบายโมเดล | ข้อ 3 |
| 4) Fairness | ข้อ 4 พร้อมปรับ threshold ดูผลกระทบได้ |
| 5) ทดลองทำนาย | กรอกข้อมูลผู้กู้ใหม่และดูผลทำนาย |

ใช้ `st.cache_data` / `st.cache_resource` เพื่อไม่ให้ฝึกโมเดลซ้ำทุกครั้งที่ผู้ใช้กดปุ่ม
(ครั้งแรกใช้เวลาประมาณ 1–2 นาที)

## 3. รันในเครื่อง

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
เปิดเบราว์เซอร์ที่ `http://localhost:8501`

## 4. อัปโหลดขึ้น GitHub

**วิธี A: ผ่านหน้าเว็บ (ไม่ต้องใช้คำสั่ง)**
1. เข้า github.com → **New repository** → ตั้งชื่อ เช่น `german-credit-streamlit` → เลือก **Public** → Create
2. กด **uploading an existing file** แล้วลากไฟล์ทั้งหมดในโฟลเดอร์นี้ขึ้นไป
   (รวมโฟลเดอร์ `data/` และ `.streamlit/`; ถ้าลากโฟลเดอร์ที่ขึ้นต้นด้วยจุดไม่ได้ ให้ใช้วิธี B หรือสร้างไฟล์
   `.streamlit/config.toml` ผ่านปุ่ม Add file → Create new file — ไฟล์นี้ไม่บังคับ)
3. กด **Commit changes**

**วิธี B: ผ่าน Git**
```bash
cd german-credit-streamlit
git init
git add .
git commit -m "German credit Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/german-credit-streamlit.git
git push -u origin main
```

## 5. Deploy บน Streamlit Community Cloud
1. เข้า https://share.streamlit.io แล้วเข้าสู่ระบบด้วยบัญชี GitHub
2. กด **Create app** → เลือก **Deploy a public app from GitHub**
3. กรอก Repository = `<username>/german-credit-streamlit`, Branch = `main`, Main file path = `app.py`
4. (ถ้ามี) เปิด **Advanced settings** เลือก Python 3.11 หรือ 3.12
5. กด **Deploy** รอติดตั้งไลบรารีจาก `requirements.txt` จากนั้นจะได้ URL ของแอป
6. เมื่อแก้โค้ดแล้ว push ขึ้น GitHub แอปจะอัปเดตอัตโนมัติ

**ปัญหาที่พบบ่อย**
- `ModuleNotFoundError` → ตรวจว่า `requirements.txt` อยู่ที่ root ของ repo
- โหลดข้อมูลจาก OpenML ไม่ได้ → แอปจะใช้ไฟล์สำรองใน `data/` อัตโนมัติ
- แอปหยุดทำงานหลังไม่มีผู้ใช้ระยะหนึ่ง → กด "Yes, get this app back up" เพื่อเปิดใหม่

## 6. ข้อจำกัด
- ช่วง hyperparameter ถูกจำกัดให้เล็กเพื่อให้รันบน Streamlit Cloud ได้ในเวลาไม่นาน
- OOF predictions ใช้ fold ชุดเดียวกับการ tune จึงอาจมองโลกในแง่ดีเล็กน้อย
  หากต้องการเข้มงวดขึ้นควรใช้ nested CV
- ชุดทดสอบมี 200 ราย กลุ่มย่อย (เช่น อายุ <25 ปี) มีขนาดเล็ก ค่า fairness จึงมีความแปรปรวนสูง
- ผลลัพธ์ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่เกณฑ์พิจารณาสินเชื่อจริง
