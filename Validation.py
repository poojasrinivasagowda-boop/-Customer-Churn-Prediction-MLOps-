from pathlib import Path
import pandas as pd
import numpy as np

# This makes sure it always saves in the right folder, even with OneDrive
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "customer_churn.csv"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(42)
    n = 2000
    df = pd.DataFrame({
        "tenure": rng.integers(1, 73, size=n),
        "monthly_charges": np.round(rng.normal(70, 20, size=n).clip(20, 130), 2),
        "support_calls": rng.poisson(1.5, size=n),
        "contract": rng.choice(["Monthly", "One Year", "Two Year"], size=n, p=[0.55, 0.25, 0.20]),
        "churn": rng.integers(0, 2, size=n)
    })
    df.to_csv(DATA_PATH, index=False)
    print(f"Data generated and saved to: {DATA_PATH}")
    print(f"Rows: {len(df)}")

if __name__ == "__main__":
    main()