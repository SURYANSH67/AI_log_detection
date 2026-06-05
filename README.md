## 📌 Project Overview

This project implements a **Hybrid Log Monitoring System** capable of:

- Detecting anomalies in infrastructure-level logs (HDFS)
- Detecting anomalies in web server logs (Apache)
- Providing severity scoring (0–100 scale)
- Generating human-readable explanations (Explainable AI)
- Visualizing anomalies via interactive dashboard
- Simulating real-time monitoring with auto-refresh

---

## 🏗️ System Architecture

Log Sources
↓
Feature Extraction
↓
Hybrid ML Engine (Isolation Forest)
↓
Severity Scoring
↓
Explainable AI Layer
↓
Streamlit Dashboard

---

## 📂 Project Structure

log_anomaly_ai/
│
├── data/
│   ├── HDFS_2k.log_structured.csv
│   ├── Apache_2k.log_structured.csv
│   └── anomaly_label.csv
│
├── src/
│   ├── step3_feature_engineering.py
│   ├── step4_train_model.py
│   ├── step5_evaluate_model.py
│   └── hybrid_system_logs.py
│
├── dashboard/
│   └── hybrid_app.py
│
├── venv/
└── README.md

---

## 🧠 Machine Learning Model

### Algorithm Used:
- **Isolation Forest (Unsupervised Learning)**

### Why Isolation Forest?
- Works without labeled data
- Effective for high-dimensional behavioral logs
- Detects anomalies via random tree isolation

---

## 📊 Features Extracted

### From HDFS (System Logs):
- Event frequency
- Unique event diversity

### From Apache (Web Logs):
- Event template frequency

These features are unified into a hybrid feature space.

---

## 🔥 Severity Scoring

Severity is calculated using:

- Distance from Isolation Forest decision boundary
- Normalized to a 0–100 scale
- Categorized as:
  - Low
  - Medium
  - High

---

## 🤖 Explainable AI (XAI)

Each anomaly includes an explanation such as:

- "Unusually high event frequency"
- "High diversity of log events"
- "Very far from normal behavior cluster"

This improves interpretability and transparency.

---

## 📈 Dashboard Features

- Real-time auto-refresh
- Anomaly count metrics
- Scatter behavior visualization
- Severity heatmap
- Ranked anomaly table
- Downloadable anomaly report (CSV)

------

## ▶️ How to Run

### 1️⃣ Activate Virtual Environment

```bash
cd log_anomaly_ai
source venv/bin/activate

2️⃣ Install Dependencies

pip install -r requirements.txt

If needed manually:

pip install streamlit pandas numpy scikit-learn matplotlib seaborn streamlit-autorefresh

3️⃣ Run Dashboard

python -m streamlit run dashboard/hybrid_app.py

Open in browser:

http://localhost:8501



🎯 Key Contributions

✔ Hybrid multi-source anomaly detection
✔ Real-time simulation
✔ Severity-based risk scoring
✔ Explainable AI integration
✔ Interactive SOC-style dashboard



🚀 Future Improvements
	•	Deep Learning (LSTM-based log modeling)
	•	SHAP-based feature importance
	•	Kafka-based real-time streaming
	•	PostgreSQL log storage
	•	Docker deployment
	•	Cloud hosting (AWS / GCP)



👨‍💻 Author

Suryansh Dixit
B.Tech CSE
AI-Based Anomaly Detection Project
