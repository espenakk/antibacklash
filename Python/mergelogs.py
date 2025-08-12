import argparse
import pandas as pd
from pathlib import Path

def merge_logs(files, out_file, testindex_col="TestIndex"):
    merged = []
    offset = 0

    for f in files:
        df = pd.read_csv(f)

        if testindex_col not in df.columns:
            raise ValueError(f"{f}: column '{testindex_col}' not found.")

        if offset > 0:
            mask = df[testindex_col] > 0
            df.loc[mask, testindex_col] = df.loc[mask, testindex_col] + offset

        merged.append(df)

        positive = df.loc[df[testindex_col] > 0, testindex_col]
        if not positive.empty:
            offset = positive.max()

    out_df = pd.concat(merged, ignore_index=True)

    out_df.to_csv(out_file, index=False)
    print(f"Wrote merged file: {out_file}")
    return out_df

def main():
    p = argparse.ArgumentParser(description="Merge log CSVs adjusting testindex.")
    p.add_argument("files", nargs="+", help="Input CSV files in desired order.")
    p.add_argument("-o","--output", default="megalog.csv", help="Output CSV filename.")
    args = p.parse_args()

    merge_logs(args.files, args.output)

if __name__ == "__main__":
    main()