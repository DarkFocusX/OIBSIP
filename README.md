# OIBSIP Internship Projects Repository

This repository contains my machine learning projects completed during the **Oasis Infobyte Internship (OIBSIP)**.  
Each project demonstrates an end-to-end workflow including data preparation, model building, evaluation, and result visualization.

## Internship Overview

The goal of this internship was to build practical, portfolio-ready data science projects by applying core machine learning concepts to real-world style problems.

- Internship Program: **OIBSIP (Oasis Infobyte Internship)**
- Domain: **Data Science / Machine Learning**
- Focus Areas: **classification, regression, NLP, EDA, model evaluation**

## Project List

1. **Iris Flower Classification**
2. **Email Spam Detection**
3. **Car Price Prediction**

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Tools / Libraries

- `nltk` (text preprocessing and stemming in spam detection)
- `joblib` (model and vectorizer persistence)
- `OneHotEncoder` and `StandardScaler` (preprocessing pipelines)
- `LinearSVC`, `LogisticRegression`, `MultinomialNB`
- `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`

## Project Descriptions

### 1) Iris Flower Classification

A beginner-friendly classification project using the famous Iris dataset.  
The project performs EDA, feature scaling, model training with KNN, and evaluation using accuracy, confusion matrix, and classification report.

**Folder:** `iris_flower_classification/`

### 2) Email Spam Detection

An NLP-based text classification project that predicts whether an SMS is **spam** or **ham**.  
It applies text cleaning, TF-IDF vectorization, and compares multiple models (Naive Bayes, Logistic Regression, Linear SVM) to select the best performer.

**Folder:** `email_spam_detection/`

### 3) Car Price Prediction

A regression project that predicts used car selling prices based on vehicle attributes.  
It includes preprocessing, feature engineering, multi-model comparison, evaluation metrics (MAE/MSE/RMSE/R2), and model saving for reuse.

**Folder:** `car_price_prediction/`

## GitHub Structure

```text
OIBSIP/
├── iris_flower_classification/
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
│   └── plots/
├── email_spam_detection/
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
│   ├── dataset/
│   ├── models/
│   └── plots/
├── car_price_prediction/
│   ├── main.py
│   ├── requirements.txt
│   ├── README.md
│   ├── dataset/
│   ├── models/
│   └── plots/
└── README.md
```

## Screenshots

Add screenshots in each project folder (recommended under `plots/`) and link them here.

### Iris Flower Classification

- Iris Class Distribution
- Iris Pairplot
- Iris Confusion Matrix

### Email Spam Detection

- Spam Label Distribution
- Spam Confusion Matrix
- Model Accuracy Comparison

### Car Price Prediction

- Price Distribution
- Correlation Heatmap
- Predicted vs Actual

> If image links do not render, run each project once to generate plots or update image paths based on your local files.

## Learning Outcomes

Through these projects, I learned how to:

- Build complete ML pipelines from data loading to evaluation
- Perform data cleaning, preprocessing, and exploratory data analysis
- Apply both classification and regression algorithms
- Work with NLP preprocessing and TF-IDF vectorization
- Compare multiple models and select the best one using metrics
- Save trained models for inference and practical use
- Document projects professionally for GitHub portfolios

## Author

**L S Rajesh**  
Machine Learning Intern - OIBSIP  
GitHub: [https://github.com/DarkFocusX](https://github.com/your-username)

---

If you like this repository, consider giving it a star.