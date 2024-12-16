import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time
import matplotlib.pyplot as plt

class GermanHousingDataProcessor:
    def __init__(self, data):
        self.raw_data = pd.DataFrame(data)
        self.processed_data = None
        self.X = None
        self.y = None

    def comprehensive_preprocessing(self):
        data = self.raw_data.copy()

        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median())

        # Create derived features
        data["property_age"] = 2024 - data["obj_yearConstructed"]

        # Encode categorical variables
        categorical_cols = data.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            data[col] = data[col].astype("category").cat.codes

        # Drop irrelevant or redundant features
        relevant_features = ["obj_livingSpace", "obj_noRooms", "property_age", "obj_purchasePrice"]
        data = data[relevant_features]

        # Split features and target
        self.X = data.drop(columns=["obj_purchasePrice"])
        self.y = data["obj_purchasePrice"]

        return self.X, self.y

    def advanced_model_training(self):
        # Preprocessing pipeline
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
            ]
        )

        # Random Forest Regressor
        pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                                   ("regressor", RandomForestRegressor(random_state=42))])

        # Hyperparameter tuning
        param_grid = {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [10, 20, 30],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4]
        }

        model = RandomizedSearchCV(pipeline, param_grid, n_iter=10, cv=3, random_state=42, scoring="neg_mean_squared_error")
        start_time = time.time()
        model.fit(self.X, self.y)
        elapsed_time = time.time() - start_time

        # Best model and cross-validation
        best_model = model.best_estimator_
        scores = cross_val_score(best_model, self.X, self.y, cv=3, scoring="neg_mean_squared_error")
        print(f"Model Training Time: {elapsed_time:.2f} seconds")
        print(f"Cross-Validation Mean MSE: {abs(np.mean(scores)):.4f}")

        return best_model

    def visualize_feature_importance(self, model):
        importances = model.named_steps["regressor"].feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = self.X.columns

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
        plt.tight_layout()
        plt.show()
