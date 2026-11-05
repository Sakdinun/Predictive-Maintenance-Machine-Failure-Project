# Predictive Maintenance: Machine Failure Classification Pipeline



โปรเจกต์พัฒนาระบบ Machine Learning เพื่อพยากรณ์และตรวจจับความเสียหายของเครื่องจักรล่วงหน้า (Predictive Maintenance) ช่วยลด Downtime ในกระบวนการผลิต และเพิ่มประสิทธิภาพการวางแผนงานซ่อมบำรุงอย่างตรงจุด



---



## Executive Summary & Business Objective

ในโรงงานอุตสาหกรรม การปล่อยให้เครื่องจักรชำรุดกะทันหันสร้างความเสียหายต่อสายการผลิตและมีต้นทุนการซ่อมแซมสูง โปรเจกต์นี้จึงมุ่งเน้นการสร้างแบบจำลองคาดการณ์ความเสี่ยงที่เครื่องจักรจะเสียหาย (`Machine failure`) โดยให้ความสำคัญกับ:

- การรับมือกับ **Imbalanced Data** (อัตราส่วนเครื่องจักรพังมีสัดส่วนน้อยมากในสภาวะปกติ)

- การป้องกัน **Data Leakage** อย่างเคร่งครัด

- การปรับ **Threshold Tuning** เพื่อให้สอดคล้องกับบริบททางธุรกิจจริง (Trade-off ระหว่างการดักจับของเสียกับการคุม False Alarm)



---



## 📊 Dataset & Preprocessing



- **Dataset Source:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)

- **Data Shape:** 10,000 แถว พร้อมตัวแปรเซนเซอร์ทางกายภาพ (อุณหภูมิ, ความเร็วรอบ, แรงบิด, การสึกหรอของหัวเครื่องมือ)

- **Data Cleansing & Leakage Prevention:**

&#x20; - กำจัดคอลัมน์ระบุตัวตน (`UDI`, `Product ID`)

&#x20; - **ตัดกลุ่มคอลัมน์ความเสียหายเฉพาะทาง (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) ออกจาก Features** เนื่องจากเป็นผลลัพธ์ย่อยหลังเกิดความเสียหาย ซึ่งอาจก่อให้เกิด Data Leakage ส่งผลให้โมเดลจำคำตอบล่วงหน้า

&#x20; - แปลงประเภทข้อมูลคุณภาพของผลิตภัณฑ์ (`Type: L, M, H`) ด้วยเทคนิค **One-Hot Encoding**

- **Data Splitting:**

&#x20; - แบ่งข้อมูลแบบ **Stratified Split** เพื่อรักษาสัดส่วนของ Class Imbalance ให้คงที่ในทุกชุดข้อมูล:

&#x20;   - **Train Set (70%):** ใช้สำหรับฝึกสอนโมเดล

&#x20;   - **Validation Set (15%):** ใช้สำหรับมอนิเตอร์ Early Stopping และจูนเกณฑ์ Threshold

&#x20;   - **Test Set (15%):** ชุดทดสอบเสมือนจริงที่ไม่เคยผ่านตาโมเดลมาก่อน (Unseen Data)



---



## 🧠 Modeling & Strategy



### 1. Algorithm & Training

###### - เลือกใช้เครื่องมือ **XGBoost Classifier** เนื่องจากทำงานร่วมกับข้อมูลตาราง (Tabular/Sensor Data) ได้อย่างมีประสิทธิภาพสูง

- **Handling Class Imbalance:** กำหนดค่า `scale_pos_weight` โดยคำนวณสัดส่วนจากเฉพาะข้อมูลชุด Train Set เพื่อเพิ่มน้ำหนักความสำคัญให้กับกรณีเครื่องจักรชำรุด (Minority Class)

- **Prevent Overfitting:** ควบคุมรอบการเทรนด้วย `early_stopping_rounds=30` ผ่าน Validation Set



### 2. Decision Strategy

- **ปลอดภัยไว้ก่อน (High Recall / Conservative):** เหมาะกับแผนการในกรณีที่ เครื่องจักรชำรุดจะสร้างความเสียหายรุนแรง จึงให้ Threshold ที่ค่อนข้างต่ำ โมเดลจึงอาจจะเตือนบ่อยขึ้น และอาจเกิดการแจ้งเตือนหลอก (Faluse Positive) มากขึ้นแต่ค่าความเสียหายน้อยกว่าไม่เตือนแล้วปล่อยให้เครื่องพังอยู่ดี

- **เกณฑ์สมดุล (Balanced Threshold):** เหมาะสำหรับแผนการซ่อมบำรุงทั่วไปที่ต้องการบริหารเวลาและภาระงานของทีมช่าง จึงปรับเกณฑ์การเตือน (Threshold) ให้พอดี เพื่อลดการแจ้งเตือนหลอก (False Positive) ไม่ให้ทีมงานเสียเวลาไปตรวจเก้อ แต่ยังคงรักษาระดับการตรวจจับความเสียหายล่วงหน้าไว้ได้ไม่ต่ำกว่า 70% เพื่อความคุ้มค่าสูงสุดในการปฏิบัติงานจริง---



## 📈 Evaluation & Results (on Unseen Test Set)



โมเดลได้รับการประเมินผลผ่านข้อมูลทดสอบ **Test Set (15%)** ที่แยกไว้ เพื่อยืนยันความสามารถในการนำไปใช้งานจริง (Generalization):

- บันทึกการประเมินผลครอบคลุมทั้ง **Precision, Recall, F1-Score** และ **Confusion Matrix**

- สามารถควบคุม False Negative (เครื่องพังจริงแต่โมเดลทายไม่เจอ) ให้อยู่ในระดับต่ำตามเป้าหมายของทีมวิศวกรรม



---



## 🛠️ Tech Stack & Libraries

- **Language:** Python

- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost

- **Environment:** Google Colab

