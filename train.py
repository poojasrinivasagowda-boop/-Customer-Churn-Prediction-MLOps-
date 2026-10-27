import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("model.pkl")

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def train():
    df = load_data()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Auto detect columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_transformed, y_train)

    preds = model.predict(X_test_transformed)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # Save model and preprocessor
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "preprocessor": preprocessor}, f)
    print(f"Model saved to {MODEL_PATH}")

def main():
    train()

if __name__ == "__main__":
    main()