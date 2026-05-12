# Iris Flower Classification (Beginner-Friendly Data Science Project)

This project is a complete beginner-friendly machine learning project using the **Iris dataset**.  
It is suitable for internship submission and GitHub portfolio upload.

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Project Workflow

The script performs all major data science steps:

1. Data loading (from `sklearn.datasets`)
2. Data exploration
3. Data visualization
4. Preprocessing (feature scaling)
5. Train-test split
6. Model training (K-Nearest Neighbors)
7. Prediction
8. Evaluation (accuracy score, confusion matrix, classification report)

## Files in This Project

- `main.py` - Complete, well-commented Python script
- `requirements.txt` - Required libraries
- `README.md` - Project documentation
- `plots/` - Automatically generated visualizations after running script

## How to Run

### 1) Create and activate virtual environment (recommended)

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the project

```bash
python main.py
```

## Output

When you run `main.py`, it will:

- Print dataset details and exploration results in terminal
- Train a KNN classification model
- Print:
  - Accuracy score
  - Classification report
  - Confusion matrix
- Save graphs in `plots/` folder:
  - `class_distribution.png`
  - `pairplot.png`
  - `correlation_heatmap.png`
  - `confusion_matrix.png`

## Why KNN?

K-Nearest Neighbors (KNN) is a beginner-friendly algorithm because:

- It is easy to understand
- It works well on small datasets like Iris
- It gives strong baseline classification performance

## Dataset Information

The Iris dataset contains 150 flower samples from 3 classes:

- Setosa
- Versicolor
- Virginica

Features used:

- Sepal length
- Sepal width
- Petal length
- Petal width

## License

This project is open for learning and educational use.
