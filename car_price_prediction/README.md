# Car Price Prediction (Beginner Friendly ML Project)

## Project Overview
This project predicts the **selling price of a used car** using machine learning regression models.  
It is designed as an internship-level, beginner-friendly, portfolio-ready project with:
- complete ML workflow
- clean project structure
- EDA visualizations
- model comparison
- saved best model
- live CLI prediction interface

---

## Project Structure

```text
car_price_prediction/
|
|-- dataset/
|-- plots/
|-- models/
|-- main.py
|-- requirements.txt
`-- README.md
```

---

## Technologies Used
- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---

## Dataset Features
The dataset includes realistic car resale attributes:
- `car_name`
- `company`
- `year`
- `fuel_type`
- `transmission`
- `kilometers_driven`
- `mileage`
- `engine`
- `horsepower`
- `seats`
- `selling_price` (target)

If `dataset/cars.csv` does not exist, `main.py` auto-generates a realistic market-inspired dataset so the project works out-of-the-box.

---

## Machine Learning Workflow
The project performs the full workflow end-to-end:

1. **Data loading**
2. **Data cleaning**
   - duplicate removal
   - datatype corrections
   - text cleanup
3. **Handling missing values**
   - numeric -> median
   - categorical -> most frequent
4. **Feature selection**
5. **Categorical encoding**
   - OneHotEncoder
6. **Train-test split**
7. **Feature scaling**
   - StandardScaler for numeric features
8. **Model training**
   - Linear Regression
   - Random Forest Regressor
   - Decision Tree Regressor
9. **Prediction & evaluation**
   - MAE
   - MSE
   - RMSE
   - R2 score
10. **Best model selection and saving**
11. **Live CLI prediction**

---

## Model Comparison
All three models are trained and compared using the same preprocessing pipeline for fair evaluation.

The script automatically:
- prints metrics for each model
- identifies the best-performing model
- saves the best model to `models/best_car_price_model.pkl`
- generates a model comparison chart

---

## Visualizations Generated (Saved in `plots/`)
1. `price_distribution.png` - selling price distribution  
2. `correlation_heatmap.png` - feature correlation heatmap  
3. `company_wise_average_price.png` - company-wise average selling price  
4. `predicted_vs_actual.png` - predicted vs actual prices (best model)  
5. `model_comparison_chart.png` - model performance comparison

---

## Setup Instructions

### 1) Clone or download project
Place the project in your local machine.

### 2) Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:
- **Windows (PowerShell):**
  ```bash
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run project

```bash
python main.py
```

---

## CLI Predictor
After model training, script asks:

`Do you want to run live CLI predictor? (yes/no)`

If you enter `yes`, you can input car details and instantly get an estimated resale price.

---

## Screenshots
Add screenshots here after running the project:
- price distribution plot
- heatmap
- predicted vs actual chart
- terminal output showing model metrics

Example markdown:

```md
![Price Distribution](plots/price_distribution.png)
![Correlation Heatmap](plots/correlation_heatmap.png)
![Predicted vs Actual](plots/predicted_vs_actual.png)
```

---

## How Regression Works (Beginner Explanation)
Regression is a supervised machine learning method used to predict **continuous values** (numbers), not categories.  
In this project, the model learns the relationship between car features (year, kilometers, engine, etc.) and selling price.  
After training, it uses these learned patterns to predict the price of a new car record.

---

## How Car Prices Are Predicted
The model observes historical car records and learns patterns such as:
- newer cars tend to cost more
- cars driven less distance usually keep better value
- higher horsepower/engine capacity can increase price
- brand/company influences resale value
- transmission and fuel type can affect demand and price

During prediction, your input goes through the same preprocessing steps and then the trained model outputs an estimated price.

---

## Why Feature Engineering Matters
Good features help the model understand real-world pricing logic better.

Examples:
- encoding text columns (`company`, `fuel_type`, etc.) into numeric format
- scaling numeric features for stable learning
- handling missing values to avoid data loss or model errors
- selecting relevant columns to reduce noise

Without proper feature engineering, model accuracy often drops significantly.

---

## Why Random Forest Often Performs Better
Random Forest is an ensemble of many decision trees. It often performs better because:
- it captures non-linear relationships
- it reduces overfitting by averaging multiple trees
- it handles mixed feature types well
- it is robust to noisy real-world data

For car pricing (which is usually non-linear and influenced by many interactions), Random Forest frequently outperforms simpler linear models.

---

## Portfolio / Internship Value
This project demonstrates:
- practical data cleaning skills
- proper ML pipeline usage
- model evaluation with multiple metrics
- model comparison and selection
- basic MLOps step (saving trained model)
- user-facing prediction interface (CLI)

It is suitable for GitHub portfolio, internship assignments, and beginner ML showcases.
