# German Credit Risk — Streamlit App (4 โมเดล)

แอปประเมินความเสี่ยงสินเชื่อจากชุดข้อมูล Statlog German Credit (OpenML `data_id=31`)
เปรียบเทียบ 4 โมเดล ได้แก่ Logistic Regression, Random Forest, Gradient Boosting และ Support Vector Machine
พร้อมการเลือก threshold ตามต้นทุน (FN:FP = 5:1), Permutation importance และ Fairness metrics

## รูปแบบการใช้งาน

| โหมด | สิ่งที่ทำได้ |
|---|---|
| **พื้นฐาน** | กรอกข้อมูลสำคัญ 8 รายการ → เกจความเสี่ยง, ผลอนุมัติ/ไม่อนุมัติ, ผลจากทั้ง 4 โมเดล, ปัจจัยที่มีผลต่อรายนั้น · สรุปผลโมเดล · คำอธิบายแอป |
| **ละเอียด** | กรอกครบ 20 ตัวแปร (เริ่มจากค่ากลางหรือตัวอย่างชุดทดสอบ) · ประเมินหลายรายจาก CSV และดาวน์โหลดผล · ปรับ threshold (จาก OOF / คำนวณจากต้นทุนใหม่ / กำหนดเอง) · EDA · ตาราง CV และ ROC curve · Permutation importance · Fairness ตามเพศและกลุ่มอายุ |

## โครงสร้าง repository

```
.
├── app.py                      # แอป Streamlit
├── train_model.py              # ฝึก 4 โมเดล (ใช้ใน Colab/เครื่อง และเป็น fallback ของแอป)
├── requirements.txt            # ใช้ไฟล์จาก Colab ทับได้ เพื่อให้เวอร์ชันตรงกัน
├── .streamlit/config.toml      # ธีมของแอป
├── german_credit.csv           # ← ได้จาก Colab
├── model_artifacts.joblib      # ← ได้จาก Colab
└── notebooks/German_Credit_Colab_v2.ipynb
```

> หากไม่มี `german_credit.csv` / `model_artifacts.joblib` หรือโหลดไม่ได้ (เช่น scikit-learn คนละเวอร์ชัน)
> แอปจะดึงข้อมูลจาก OpenML และฝึกโมเดลใหม่อัตโนมัติ (ประมาณ 1 นาทีในการเปิดครั้งแรก)

## ขั้นตอน deploy

1. **Colab** — เปิด `notebooks/German_Credit_Colab_v2.ipynb` → Runtime → Run all → ได้ `deploy_files.zip`
2. **GitHub** — สร้าง repository แบบ Public → Add file → Upload files
   - อัปโหลดไฟล์ทั้งหมดในโฟลเดอร์นี้ (รวมโฟลเดอร์ `.streamlit`)
   - แตก `deploy_files.zip` แล้วอัปโหลด `german_credit.csv`, `model_artifacts.joblib`, `requirements.txt` ไว้ที่ root (ทับไฟล์เดิม)
   - หมายเหตุ: การลากไฟล์ผ่านเว็บอาจข้ามโฟลเดอร์ที่ขึ้นต้นด้วยจุด หากไม่เห็น `.streamlit/config.toml` ให้กด Add file → Create new file แล้วพิมพ์ชื่อ `.streamlit/config.toml` และวางเนื้อหา (ไม่มีไฟล์นี้แอปก็ทำงานได้ เพียงใช้ธีมเริ่มต้น)
3. **Streamlit Community Cloud** — https://share.streamlit.io → Create app → เลือก repository, branch `main`, main file `app.py`
   → Advanced settings → Python ให้ตรงกับ Colab (ตรวจด้วย `!python --version`) → Deploy

## รันบนเครื่อง

```bash
pip install -r requirements.txt
python train_model.py      # สร้าง german_credit.csv และ model_artifacts.joblib (ถ้ายังไม่มี)
streamlit run app.py
```

## ข้อควรทราบ

- ไฟล์ `.joblib` ต้องโหลดด้วย scikit-learn เวอร์ชันเดียวกับตอนบันทึก จึงต้องใช้ `requirements.txt` ที่ Colab สร้าง
- ผลการประเมินใช้เพื่อการศึกษา ไม่ควรใช้ตัดสินสินเชื่อจริงโดยลำพัง
