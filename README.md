UPI Fraud Detection System (Machine Learning + Risk Engine + Real-Time Monitoring)

This project is an end-to-end UPI Fraud Detection & Risk Management System built with Python, Machine Learning, and Real-Time Monitoring mechanisms.
It includes:

✅ Synthetic UPI dataset generation

✅ Data preprocessing + feature engineering

✅ Multiple ML models (RF, GBM, Logistic Regression)

✅ Feature importance visualization

✅ Fraud probability scoring

✅ Risk classification (Low/Medium/High/Critical)

✅ Real-time transaction monitoring using threading

✅ Modular, scalable class-based design



📌 Project Overview


The goal of this project is to simulate a complete UPI fraud detection pipeline similar to systems used by banks, payment apps, and fintech companies.

The system includes:

1️⃣ UPIFraudDetector

Generates synthetic transaction data

Preprocesses categorical & numerical features

Performs feature engineering

Trains 3 ML models:

Random Forest

Gradient Boosting

Logistic Regression

Computes feature importance & evaluation metrics



2️⃣ RiskManagementSystem

Uses the trained ML model to compute fraud probability

Assigns a risk level:

LOW → Approve

MEDIUM → Review

HIGH → Ask for 2FA

CRITICAL → Block

Returns fraud probability + recommended action



3️⃣ RealTimeUPIMonitor

A live monitoring engine that:

Accepts incoming transactions

Runs fraud prediction in real-time

Generates alerts when risk >= HIGH

Logs every transaction with timestamp

Runs in a background thread to mimic real UPI systems



4️⃣ AdvancedUPIFraudDetector (Optional)

(If XGBoost is available)

Builds a Voting Ensemble

Tunes dynamic thresholds using a business-aware scoring function


📁 Folder Structure (recommended)
│── UPIFraudDetection/
│   ├── fraud_detector.py
│   ├── risk_management.py
│   ├── real_time_monitor.py
│   ├── advanced_detector.py
│   ├── main.py
│   ├── README.md
│   └── requirements.txt


⚙️ Installation

1. Clone the repository
git clone https://github.com/your-username/UPI-Fraud-Detection.git
cd UPI-Fraud-Detection

2. Install dependencies
Create requirements.txt with:

pandas
numpy
matplotlib
scikit-learn
seaborn
xgboost



Then run:

pip install -r requirements.txt

▶️ How to Run the Project
Run the full pipeline

This executes:

✔ Data generation
✔ Model training
✔ Evaluation
✔ Feature importance plots
✔ Real-time risk assessment
✔ Real-time transaction monitoring

python main.py

📊 Synthetic Dataset

Each transaction contains fields such as:

amount

time_of_day

location_city

transaction_type

device_type

user_income_bracket

previous_chargebacks

transaction frequency

risk-based engineered features

Fraud is injected using:

high amount

foreign location

new device

night-time transactions

high frequency

past chargebacks

🧠 Machine Learning Models Used

Model	Purpose	Notes:-
Random Forest	Primary fraud classifier	Works without scaling
Gradient Boosting	Risk scoring	Tree boosting
Logistic Regression	Extra light-weight classifier	Uses scaled data
Voting Ensemble	(Optional) Combines RF + GB + XGB	Best accuracy

Evaluation metrics include:

AUC Score

Classification Report

Confusion Matrix

📈 Feature Importance Visualization

The project automatically plots the Top 15 most important features for tree-based models such as Random Forest and Gradient Boosting.


🛡️ Risk Engine Logic

Risk levels:

Probability	Risk Level	Action
< 0.30	LOW	APPROVE
< 0.70	MEDIUM	REVIEW
< 0.90	HIGH	REQUIRE 2FA
>= 0.90	CRITICAL	BLOCK
🔴 Real-Time Transaction Monitoring

Simulates real-time UPI transactions using:

threading.Thread

deque for buffering

Automated alerts

Example alert:

ALERT: HIGH RISK - 2FA required: 5123.78


Every transaction is logged as:

{
  "timestamp": 1732600000.123,
  "transaction_id": 10005,
  "amount": 4500.5,
  "risk_level": "HIGH",
  "fraud_probability": 0.82,
  "action_taken": "2FA_REQUIRED"
}

⭐ Key Features

✔ Machine Learning-based fraud prediction

✔ Modular class design

✔ Real-time monitoring

✔ Feature engineering

✔ Risk-based transaction routing

✔ Ensemble support (XGBoost optional)

✔ Automatically generated synthetic UPI dataset


📌 Usage Examples
Single Transaction Risk Assessment
risk_system.assess_risk({
    "amount": 3500,
    "time_of_day": 2,
    "device_type": "Mobile",
    "is_new_device": 1,
    ...
})

📄 License

This project is open-source under the MIT License.

🙌 Contributions

Pull requests are welcome!
You can improve:

Dataset realism

Deep learning models (LSTM/Transformers)

Dashboard UI (Streamlit)

Integration with Kafka/WebSockets
