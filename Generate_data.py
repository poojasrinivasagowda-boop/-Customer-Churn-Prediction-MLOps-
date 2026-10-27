from pathlib import Path
import numpy as np
import pandas as pd
SEED = 42
ROWS = 2000

def main():
    rng = np.random.default_rng(SEED)
    tenure = rng.integers(1, 73, size=ROWS)
    monthly_charges = np.round(
        rng.normal(70, 20, size=ROWS).clip(20, 130), 2
    )
    support_calls = rng.poisson(1.5, size=ROWS)
    contract = rng.choice(
        ["Monthly", "One Year", "Two Year"],
        size=ROWS,
        p=[0.55, 0.25, 0.20],
    )
    # ... rest of your code
    print("Data generated")

if __name__ == "__main__":
    main()