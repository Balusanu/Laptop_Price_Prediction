# 💻 Laptop Price Prediction Using Linear Regression

Predict laptop prices based on configuration using a Linear Regression model. This project demonstrates a full ML workflow from data collection to deployment using Streamlit.

---

## **Project Workflow**

1. **Data Collection & Loading** – Gathered and imported the laptop dataset for analysis.
2. **Exploratory Data Analysis (EDA)** – Visualized data patterns, relationships, and insights using plots and summary statistics.
3. **Data Cleaning & Preprocessing** – Handled missing values, duplicates, and ensured consistent data formats.
4. **Feature Engineering & Selection** – Created meaningful features and selected the most relevant predictors.
5. **Train-Test Split** – Divided data into training and testing sets for model validation.
6. **Model Training** – Trained a Linear Regression model on the training data.
7. **Model Testing** – Made predictions on the test set and assessed generalization.
8. **Model Evaluation** – Evaluated performance using R², MAE, RMSE, and cross-validation.
9. **Model Deployment** – Deployed the trained model using **Streamlit** for interactive user predictions.

---

## **Dataset Columns**

| Column               | Description                                 |
| -------------------- | ------------------------------------------- |
| **Brand**            | Laptop brand (Dell, HP, Lenovo, Asus, Acer) |
| **Processor_Speed**  | Processor speed in GHz (1.5–4.0)            |
| **RAM_Size**         | RAM size in GB (4, 8, 16, 32)               |
| **Storage_Capacity** | Storage in GB (256, 512, 1000)              |
| **Screen_Size**      | Screen size in inches (11–17)               |
| **Weight**           | Laptop weight in kg (2–5)                   |
| **Price**            | Target variable – laptop price              |

---

## **Libraries Used**

* **NumPy & Pandas** – Data manipulation and analysis
* **Matplotlib & Seaborn** – Data visualization and EDA
* **Scikit-Learn (sklearn)** – Linear Regression modeling, scaling, and evaluation
* **Streamlit & Pickle** – Model deployment and saving/loading artifacts

---

## **Data Preprocessing & Insights**

* **Categorical Columns:** Brand, RAM_Size, Storage_Capacity

* **Numerical Columns:** Processor_Speed, Screen_Size, Weight

* **Target Column:** Price

* No null or duplicate values detected.

* One-hot encoding applied for Brand column.

* Distribution of Price visualized via Boxplots.

* Relation between Price and numerical/categorical columns analyzed using Scatterplots and bar plots.

**Correlation with Price:**

| Feature              | Correlation | Interpretation                                                     |
| -------------------- | ----------- | ------------------------------------------------------------------ |
| **Storage_Capacity** | 0.9979      | Extremely high positive correlation – strongest predictor of Price |
| **RAM_Size**         | 0.0612      | Weak positive correlation – small effect on price                  |
| **Weight**           | 0.0384      | Very weak correlation – negligible impact                          |
| **Processor_Speed**  | -0.0507     | Very weak negative correlation                                     |
| **Screen_Size**      | -0.0267     | Very weak negative correlation                                     |
| **Brand_***          | ~±0.02      | Weak correlation individually                                      |

* **Weight column removed** as it has minimal impact.
* No strong multicollinearity among numeric predictors.

---

## **Model Evaluation**

| Metric                         | Value            |
| ------------------------------ | ---------------- |
| Mean Absolute Error (MAE)      | 147.61           |
| Mean Squared Error (MSE)       | 33,137.48        |
| Root Mean Squared Error (RMSE) | 182.04           |
| R² Score                       | 0.9996           |
| Adjusted R² Score              | 0.9996           |
| Train R²                       | 0.9995           |
| Train RMSE                     | 40,002.88        |
| Test R²                        | 0.9996           |
| Test RMSE                      | 33,137.48        |
| Cross-validation R²            | 0.9995 (average) |

* Residuals show constant variance and normal distribution.
* Model performs well on both train and test sets – no overfitting detected.

---

## **Deployment Using Streamlit**

* **Interactive Web App**: Users can select laptop configuration (Brand, RAM, Storage, Processor Speed, Screen Size) and predict the price instantly.
* **Artifacts Saved**: Model, scaler, and metadata saved using **pickle**.
* **One-Hot Encoding & Scaling**: Applied dynamically in the app to match training data features.
* **Download Option**: Users can download the prediction results as a CSV file.

**Live Demo:** [Click here to try the app](https://laptoppriceprediction-ju8yzbkw7els2tzun53szz.streamlit.app/)

---

## **Folder Structure**

```
Laptop_Price_Prediction/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── artifacts/             # Saved model, scaler, meta
│   ├── model.pkl
│   ├── scalar.pkl
│   └── meta.pkl
└── README.md              # Project documentation
```

---

## **How to Run Locally**

1. Clone the repo:

```bash
git clone https://github.com/<your-username>/laptop-price-predictor.git
cd laptop-price-predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## **Conclusion & Insights**

* **Storage capacity** is the strongest predictor of laptop price.
* Linear Regression performs exceptionally well with minimal error.
* The deployed app provides **real-time predictions** with an intuitive UI for users.
* The project demonstrates a **full ML workflow** from preprocessing, training, evaluation, to deployment.
