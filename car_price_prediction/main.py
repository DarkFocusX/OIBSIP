"""
Car Price Prediction Project (Beginner Friendly)
------------------------------------------------
This script creates a full machine learning workflow:
1) Loads (or creates) a realistic car dataset
2) Cleans and prepares data
3) Performs exploratory data analysis (EDA)
4) Trains and compares multiple regression models
5) Saves visualizations and best model
6) Starts an interactive CLI predictor
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(".")
DATASET_DIR = BASE_DIR / "dataset"
PLOTS_DIR = BASE_DIR / "plots"
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = DATASET_DIR / "cars.csv"
MODEL_PATH = MODELS_DIR / "best_car_price_model.pkl"


def ensure_project_dirs() -> None:
    """Create the required folders if they do not exist."""
    for folder in [DATASET_DIR, PLOTS_DIR, MODELS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def create_realistic_dataset(path: Path, n_samples: int = 320) -> None:
    """
    Create a realistic synthetic used-car dataset.

    Why synthetic?
    - It keeps this project self-contained and easy to run for beginners.
    - Values are generated from practical market-inspired ranges.
    """
    np.random.seed(42)

    car_profiles = [
        ("Swift", "Maruti", 1197, 82, 5),
        ("Baleno", "Maruti", 1197, 89, 5),
        ("City", "Honda", 1498, 119, 5),
        ("Amaze", "Honda", 1199, 89, 5),
        ("i20", "Hyundai", 1197, 88, 5),
        ("Creta", "Hyundai", 1497, 113, 5),
        ("Innova", "Toyota", 2393, 148, 7),
        ("Fortuner", "Toyota", 2755, 201, 7),
        ("Nexon", "Tata", 1199, 118, 5),
        ("Harrier", "Tata", 1956, 168, 5),
        ("XUV700", "Mahindra", 1999, 197, 7),
        ("Scorpio", "Mahindra", 2184, 130, 7),
        ("Seltos", "Kia", 1497, 113, 5),
        ("Sonet", "Kia", 1197, 82, 5),
        ("Polo", "Volkswagen", 999, 108, 5),
        ("Verna", "Hyundai", 1497, 113, 5),
    ]

    fuel_choices = ["Petrol", "Diesel", "CNG"]
    trans_choices = ["Manual", "Automatic"]

    rows = []
    for _ in range(n_samples):
        car_name, company, engine_base, hp_base, seats_base = car_profiles[np.random.randint(0, len(car_profiles))]
        year = np.random.randint(2008, 2025)
        car_age = 2026 - year
        fuel_type = np.random.choice(fuel_choices, p=[0.60, 0.33, 0.07])
        transmission = np.random.choice(trans_choices, p=[0.72, 0.28])

        kilometers_driven = np.random.randint(10000, 220000)
        mileage = np.random.normal(18, 3.8)
        if fuel_type == "Diesel":
            mileage += 1.3
        if fuel_type == "CNG":
            mileage += 2.2
        mileage = float(np.clip(mileage, 10, 32))

        engine = int(np.clip(np.random.normal(engine_base, 120), 800, 3000))
        horsepower = int(np.clip(np.random.normal(hp_base, 15), 55, 280))
        seats = seats_base if np.random.rand() > 0.15 else np.random.choice([4, 5, 6, 7])

        # Market-inspired pricing formula with noise
        base_price = 220000
        company_factor = {
            "Maruti": 0.95,
            "Hyundai": 1.00,
            "Honda": 1.05,
            "Tata": 1.03,
            "Mahindra": 1.18,
            "Toyota": 1.35,
            "Kia": 1.10,
            "Volkswagen": 1.12,
        }[company]
        fuel_bonus = {"Petrol": 0, "Diesel": 50000, "CNG": 20000}[fuel_type]
        trans_bonus = 65000 if transmission == "Automatic" else 0

        price = (
            base_price
            + (year - 2008) * 30000
            + horsepower * 6000
            + engine * 120
            + fuel_bonus
            + trans_bonus
            + seats * 8000
            - kilometers_driven * 1.6
            - car_age * 45000
        ) * company_factor

        noise = np.random.normal(0, 90000)
        selling_price = max(120000, int(price + noise))

        rows.append(
            {
                "car_name": car_name,
                "company": company,
                "year": year,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "kilometers_driven": kilometers_driven,
                "mileage": round(mileage, 2),
                "engine": engine,
                "horsepower": horsepower,
                "seats": seats,
                "selling_price": selling_price,
            }
        )

    df = pd.DataFrame(rows)

    # Add a few missing values intentionally to demonstrate handling.
    for col in ["mileage", "engine", "horsepower", "fuel_type"]:
        miss_idx = np.random.choice(df.index, size=max(1, int(0.02 * len(df))), replace=False)
        df.loc[miss_idx, col] = np.nan

    df.to_csv(path, index=False)


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load dataset and clean data types, duplicates, and missing values."""
    df = pd.read_csv(path)

    # Remove exact duplicate rows.
    df = df.drop_duplicates().reset_index(drop=True)

    # Ensure numeric columns are numeric.
    numeric_cols = ["year", "kilometers_driven", "mileage", "engine", "horsepower", "seats", "selling_price"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean text columns.
    text_cols = ["car_name", "company", "fuel_type", "transmission"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()

    # Handle missing values for each column type.
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in text_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def create_eda_plots(df: pd.DataFrame) -> None:
    """Create and save all required EDA visualizations."""
    sns.set_theme(style="whitegrid")

    # 1) Price distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["selling_price"], bins=30, kde=True, color="royalblue")
    plt.title("Selling Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "price_distribution.png", dpi=160)
    plt.close()

    # 2) Correlation heatmap
    plt.figure(figsize=(9, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    # 3) Company-wise average price
    company_avg = df.groupby("company", as_index=False)["selling_price"].mean().sort_values("selling_price", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=company_avg, x="company", y="selling_price", palette="viridis")
    plt.title("Company-wise Average Selling Price")
    plt.xlabel("Company")
    plt.ylabel("Average Price")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "company_wise_average_price.png", dpi=160)
    plt.close()


def build_model_pipeline(model) -> Pipeline:
    """Create preprocessing + model pipeline."""
    categorical_features = ["car_name", "company", "fuel_type", "transmission"]
    numeric_features = ["year", "kilometers_driven", "mileage", "engine", "horsepower", "seats"]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate_model(model_name: str, pipeline: Pipeline, x_train, x_test, y_train, y_test):
    """Train model, predict, and return metrics + predictions."""
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": preds,
    }


def save_prediction_plot(y_test, preds, model_name: str) -> None:
    """Save predicted vs actual plot for the best model."""
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_test, y=preds, alpha=0.7, color="teal")
    min_v = min(float(np.min(y_test)), float(np.min(preds)))
    max_v = max(float(np.max(y_test)), float(np.max(preds)))
    plt.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--", linewidth=2)
    plt.title(f"Predicted vs Actual Prices ({model_name})")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "predicted_vs_actual.png", dpi=160)
    plt.close()


def save_model_comparison_chart(results_df: pd.DataFrame) -> None:
    """Save model comparison chart for R2 score."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="Model", y="R2", palette="mako")
    plt.title("Model Comparison (R2 Score)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison_chart.png", dpi=160)
    plt.close()


def run_cli_predictor(best_model: Pipeline, reference_df: pd.DataFrame) -> None:
    """Simple live CLI where user enters details and gets predicted car price."""
    print("\n--- Live Car Price Predictor ---")
    print("Enter car details to estimate selling price.")
    print("Tip: Use values close to your real car for better estimates.\n")

    unique_car_names = sorted(reference_df["car_name"].unique())
    unique_companies = sorted(reference_df["company"].unique())
    unique_fuels = sorted(reference_df["fuel_type"].unique())
    unique_transmissions = sorted(reference_df["transmission"].unique())

    print(f"Available car names: {', '.join(unique_car_names)}")
    car_name = input("Car name: ").strip()

    print(f"Available companies: {', '.join(unique_companies)}")
    company = input("Company: ").strip()

    year = int(input("Year (e.g., 2018): ").strip())

    print(f"Available fuel types: {', '.join(unique_fuels)}")
    fuel_type = input("Fuel type: ").strip()

    print(f"Available transmission types: {', '.join(unique_transmissions)}")
    transmission = input("Transmission: ").strip()

    kilometers_driven = float(input("Kilometers driven: ").strip())
    mileage = float(input("Mileage (km/l): ").strip())
    engine = float(input("Engine (CC): ").strip())
    horsepower = float(input("Horsepower (bhp): ").strip())
    seats = float(input("Number of seats: ").strip())

    input_df = pd.DataFrame(
        [
            {
                "car_name": car_name,
                "company": company,
                "year": year,
                "fuel_type": fuel_type,
                "transmission": transmission,
                "kilometers_driven": kilometers_driven,
                "mileage": mileage,
                "engine": engine,
                "horsepower": horsepower,
                "seats": seats,
            }
        ]
    )

    predicted_price = best_model.predict(input_df)[0]
    print(f"\nEstimated Selling Price: Rs. {predicted_price:,.2f}")
    print("Note: Prediction is an estimate based on training data patterns.\n")


def main() -> None:
    """Run the complete beginner-friendly ML workflow."""
    ensure_project_dirs()

    if not DATASET_PATH.exists():
        create_realistic_dataset(DATASET_PATH)
        print(f"Dataset created at: {DATASET_PATH}")

    print("Loading dataset...")
    df = load_and_clean_data(DATASET_PATH)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nCreating EDA plots...")
    create_eda_plots(df)

    # Feature selection: choose useful input columns and target column.
    features = [
        "car_name",
        "company",
        "year",
        "fuel_type",
        "transmission",
        "kilometers_driven",
        "mileage",
        "engine",
        "horsepower",
        "seats",
    ]
    target = "selling_price"

    x = df[features]
    y = df[target]

    print("\nSplitting data into train and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    }

    all_results = []
    print("\nTraining and evaluating models...\n")
    for model_name, model in models.items():
        pipeline = build_model_pipeline(model)
        result = evaluate_model(model_name, pipeline, x_train, x_test, y_train, y_test)
        all_results.append(result)

        print(f"{model_name}")
        print(f"MAE  : {result['mae']:.2f}")
        print(f"MSE  : {result['mse']:.2f}")
        print(f"RMSE : {result['rmse']:.2f}")
        print(f"R2   : {result['r2']:.4f}\n")

    # Select best model using highest R2 score (primary metric).
    all_results_sorted = sorted(all_results, key=lambda r: (r["r2"], -r["rmse"]), reverse=True)
    best_result = all_results_sorted[0]
    best_model = best_result["pipeline"]

    print("=" * 60)
    print(f"Best Model: {best_result['model_name']}")
    print(f"Best R2 Score: {best_result['r2']:.4f}")
    print("=" * 60)

    # Save best model
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved to: {MODEL_PATH}")

    # Save required plots
    save_prediction_plot(y_test, best_result["predictions"], best_result["model_name"])
    comparison_df = pd.DataFrame(
        [{"Model": r["model_name"], "MAE": r["mae"], "MSE": r["mse"], "RMSE": r["rmse"], "R2": r["r2"]} for r in all_results]
    ).sort_values("R2", ascending=False)
    save_model_comparison_chart(comparison_df)

    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))
    print("\nAll plots saved inside 'plots/' directory.")

    # Start interactive predictor only if user wants.
    run_predictor = input("\nDo you want to run live CLI predictor? (yes/no): ").strip().lower()
    if run_predictor in {"yes", "y"}:
        run_cli_predictor(best_model, df)
    else:
        print("CLI predictor skipped.")


if __name__ == "__main__":
    main()
