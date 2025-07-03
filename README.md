
# 📈 Customer Lifetime Value Prediction with BG/NBD Model

An interactive Streamlit app to analyze customer transaction data and predict **Customer Lifetime Value (CLV)** using the **BG/NBD model** from the Lifetimes library. Users can upload their sales data, transform it, train a model, visualize results, and download predictions with model parameters.

---

## 🔍 Features

- 📁 Upload or generate sample transaction data
- 🧼 Data validation and preprocessing
- 📊 Automated feature engineering using `summary_data_from_transaction_data`
- 🤖 Train BG/NBD model with customizable penalizer coefficient
- 📈 Predict customer purchases over any future time horizon
- 🔥 Visualize frequency-recency matrix and model validation curve
- 💾 Download predictions and model parameters as a ZIP archive

---

## 🧱 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Lifetimes (BG/NBD model)](https://github.com/CamDavidsonPilon/lifetimes)
- [Altair](https://altair-viz.github.io/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas, NumPy, JSON, Zipfile]

---

## ⚙️ Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/your-username/clv-prediction-app.git
cd clv-prediction-app
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run your_script.py
```

---

## 📋 Required Input Format

The uploaded CSV file must include the following columns:

- `customer_id`: Unique customer identifier
- `date`: Date of transaction (`YYYY-MM-DD` format recommended)

---

## 🧪 How It Works

1. Upload a sales history file or generate a sample.
2. Data is validated and processed.
3. A BG/NBD model is trained using lifetimes.
4. Predictions are made over a selected time period.
5. Visualizations show purchasing behavior and model fit.
6. Results can be downloaded as a `.zip` file.

---

## 📁 Output

- `predictions.csv`: Customer-level predicted transaction counts
- `model_params.json`: Trained model parameters

---

## 📄 License

MIT License — use, modify, and share freely.

