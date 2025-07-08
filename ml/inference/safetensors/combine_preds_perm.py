import os
import glob
import pandas as pd

def combine_predictions(input_dir: str, output_csv: str):
    """
    Finds all *_preds.csv in input_dir, extracts model names,
    and merges their 'prediction' columns (grouped by permeability)
    into one CSV.
    """
    # 1) find all matching CSVs
    pattern = os.path.join(input_dir, "*_preds.csv")
    csv_paths = sorted(glob.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching '*_preds.csv' in {input_dir}")

    combined_df = None

    for path in csv_paths:
        # 2) extract model name (prefix before "_preds.csv")
        model_name = os.path.basename(path).rsplit("_preds.csv", 1)[0]
        print(model_name)

        # 3) read only the 2nd (permeability) and 3rd (prediction) columns by position
        #    header=0 assumes the first row is the header; adjust if your files differ
        df = pd.read_csv(path, usecols=[1, 2], header=0)
        df.columns = ["permeability", model_name]

        # 4) if the same permeability appears multiple times, average the predictions
        # df = df.groupby("permeability", as_index=False).mean()

        # 5) merge into the running combined_df
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on="permeability", how="outer")

    # 6) (optional) drop any permeability rows missing a model’s prediction
    combined_df = combined_df.dropna()

    # 7) save
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined predictions written to: {output_csv}")


if __name__ == "__main__":
    input_dir = '/home/josh/clotsimnet/data/preds'
    
    out_csv = 'combined_preds2.csv'
    
    combine_predictions(input_dir=input_dir, output_csv=out_csv)
