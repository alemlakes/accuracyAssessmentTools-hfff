import pandas as pd
import os

def trim_probs(input_path, output_path, prob_cols):
    df = pd.read_csv(input_path)
    for col in prob_cols:
        if col in df.columns:
            df[col] = df[col].round(3)
    df.to_csv(output_path, index=False)

# File paths
base_dir = os.path.dirname(__file__)
raw_dir = os.path.join(base_dir, '1.rawOutputs')
trimmed_dir = os.path.join(base_dir, '2.trimmedOutputs')
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(trimmed_dir, exist_ok=True)
files = [
    ("1.hfff_binary.csv", "1b.hfff_binary.csv", ["Growth", "Loss", "Stable"]),
    ("2.hfff_cwci.csv", "2b.hfff_cwci.csv", ["Growth", "Loss", "Stable"]),
    ("3.hfff_discount.csv", "3b.hfff_discount.csv", ["Growth", "Loss", "Stable"]),
    ("4.hfff_composite.csv", "4b.hfff_composite.csv", [
        "Growth", "Loss", "Stable",
        "Growth_cwci", "Loss_cwci", "Stable_cwci",
        "Growth_discount", "Loss_discount", "Stable_discount"
    ]),
]

for in_file, out_file, cols in files:
    trim_probs(os.path.join(raw_dir, in_file), os.path.join(trimmed_dir, out_file), cols)
