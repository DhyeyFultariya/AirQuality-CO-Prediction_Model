# Air Quality CO(GT) Prediction Model

## 📊 Dataset
Air Quality dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality).

## 🧠 Problem Statement
Predict the concentration of carbon monoxide (CO(GT)) using sensor readings and other environmental features.

## 🧪 Features Used
The following engineered and selected features were used in the final model:
- `CO_rolling_mean_3`
- `CO_lag_1`
- `C6H6(GT)`
- `PT08.S2(NMHC)`
- `NOx(GT)`
- `PT08.S1(CO)`
- `PT08.S5(O3)`
- `NO2(GT)`
- `CO_lag_2`
- `PT08.S4(NO2)`
- `CO_rolling_std_3`
- `Hour`

## 🛠️ Steps
- Data Cleaning (`-200` values replaced/dropped)
- Feature Engineering (lags, rolling means)
- Correlation-based feature selection
- Model Training using Random Forest Regressor
- Evaluation using MAE, RMSE, R²

## 🔧 Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## ⚙️ Model
The final model used was:
- **Random Forest Regressor**
- Evaluation Metric: R² Score

## 📈 Results
- **R² Score**: ~0.97 on the test dataset

## 📁 Files
- `Final_Model.ipynb`: Jupyter Notebook with full preprocessing, feature engineering, and model training
- `requirements.txt`: Dependencies to install
- `model.pkl`: Trained model file for inference
- `Dashboard.py`: Dashboard code, Use streamlit for generate Dashboard. For show dashboard, Run "streamlit run Dashboard.py" in terminal.
- [View Dashboard](https://airquality-co-predictionmodel-tesk9xly3yvu3pnql6mk6b.streamlit.app)
---

Made with ❤️ for learning and research.

