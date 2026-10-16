from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "Mall_Customers_synthetic.csv"
report_path = base_dir / "mall_customers_profile.html"

# Load the dataset
df = pd.read_csv(csv_path)

# Generate the data profiling report
profile = ProfileReport(
    df,
    title="Mall Customers Synthetic Profiling Report",
    explorative=True,
)

# Export to HTML
profile.to_file(report_path)

print(f"Dataset loaded from: {csv_path}")
print(f"Dataset shape: {df.shape}")
print(f"Profiling report saved to: {report_path}")