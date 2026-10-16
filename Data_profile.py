from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "Mall_Customers_synthetic.csv"

df = pd.read_csv(csv_path)

# Force numeric columns to a numeric dtype so comparisons remain consistent across splits
for col in ["CustomerID", "Age", "Annual Income (k$)", "Spending Score (1-100)"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Shuffle and split the dataset into train/test sets
randomized_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_index = int(len(randomized_df) * 0.8)
train_df = randomized_df.iloc[:split_index].copy()
test_df = randomized_df.iloc[split_index:].copy()

train_profile = ProfileReport(
    train_df,
    title="Training Data Profiling Report",
    explorative=True,
)
test_profile = ProfileReport(
    test_df,
    title="Testing Data Profiling Report",
    explorative=True,
)

# Compare the profiles of train and test data
comparison_profile = train_profile.compare(test_profile)

# Save outputs
train_profile.to_file(base_dir / "train_profile.html")
test_profile.to_file(base_dir / "test_profile.html")
comparison_profile.to_file(base_dir / "train_test_comparison.html")

print(f"Dataset loaded from: {csv_path}")
print(f"Total rows: {len(df)}")
print(f"Training rows: {len(train_df)}")
print(f"Testing rows: {len(test_df)}")
print(f"Saved training report: {base_dir / 'train_profile.html'}")
print(f"Saved testing report: {base_dir / 'test_profile.html'}")
print(f"Saved comparison report: {base_dir / 'train_test_comparison.html'}")

# Quick numeric comparison summary
print("\nQuick numeric comparison:")
for col in df.select_dtypes(include=["number"]).columns:
    train_mean = train_df[col].mean()
    test_mean = test_df[col].mean()
    train_std = train_df[col].std(ddof=1)
    test_std = test_df[col].std(ddof=1)
    print(
        f"- {col}: train mean={train_mean:.2f}, test mean={test_mean:.2f}, "
        f"train std={train_std:.2f}, test std={test_std:.2f}"
    )