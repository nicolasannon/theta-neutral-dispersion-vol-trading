from pathlib import Path
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

ROOT = Path("data")

def inspect_csv(path: Path):
    print(f"\n=== CSV: {path} ===")
    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes)
    print("head:")
    print(df.head(5))
    print("nulls:")
    print(df.isna().sum())

def inspect_parquet(path: Path, n=5):
    print(f"\n=== PARQUET: {path} ===")
    if pq is not None:
        pf = pq.ParquetFile(path)
        print("num_row_groups:", pf.num_row_groups)
        print("schema:")
        print(pf.schema_arrow)
    df = pd.read_parquet(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes)
    print("head:")
    print(df.head(n))
    print("nulls (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

if __name__ == "__main__":
    inspect_csv(ROOT / "par-yield-curve-rates-2020-2023.csv")
    inspect_parquet(ROOT / "aapl_2016_2023.parquet")
    inspect_parquet(ROOT / "spy_2020_2022.parquet" / "part.0.parquet")

    optdb_root = ROOT / "optiondb_2016_2023.parquet"
    first_date_dir = sorted([p for p in optdb_root.iterdir() if p.is_dir()])[0]
    first_file = next(first_date_dir.glob("*.parquet"))
    print("\nFirst optiondb partition folder:", first_date_dir.name)
    inspect_parquet(first_file)