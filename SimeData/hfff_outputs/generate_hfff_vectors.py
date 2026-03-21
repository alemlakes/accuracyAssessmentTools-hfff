import pandas as pd
import numpy as np
import json
import os
from acc_assessment.standardizer import ProbStandardizer

def parse_json_field(field):
    try:
        return json.loads(field.replace("'", '"'))
    except Exception:
        return {}

def get_hinder_value(json_str):
    d = parse_json_field(json_str)
    hinder = d.get('Hinder') or d.get('Hinder'.lower())
    if hinder:
        return float(hinder.replace('%',''))
    return 0.0

def get_helped_value(json_str):
    d = parse_json_field(json_str)
    helped = d.get('Helped') or d.get('Helped'.lower())
    if helped:
        return float(helped.replace('%',''))
    return 0.0

def binary_heuristic_thresholding(row, threshold=50):
    hinders = [get_hinder_value(row['Image_Condition']), get_hinder_value(row['Spatial_Context']), get_hinder_value(row['Spectral_Capacity'])]
    max_hinder = max(hinders)
    is_confident = max_hinder < threshold
    # Dominant class: highest of Rec_Prob_12month_* columns
    probs = [row['Rec_Prob_12month_Growth'], row['Rec_Prob_12month_Loss'], row['Rec_Prob_12month_Stable']]
    class_names = ['Growth', 'Loss', 'Stable']
    dominant_idx = int(np.argmax(probs))
    return class_names[dominant_idx], is_confident

def continuous_weighted_confidence(row):
    helped = [get_helped_value(row['Image_Condition']), get_helped_value(row['Spatial_Context']), get_helped_value(row['Spectral_Capacity'])]
    hinder = [get_hinder_value(row['Image_Condition']), get_hinder_value(row['Spatial_Context']), get_hinder_value(row['Spectral_Capacity'])]
    net = [(h - hi)/100 for h, hi in zip(helped, hinder)]
    avg_net = np.mean(net)
    # Normalize from [-1,1] to [0,1]
    cwci = (avg_net + 1) / 2
    # Dominant class: highest of Rec_Prob_12month_* columns
    probs = [row['Rec_Prob_12month_Growth'], row['Rec_Prob_12month_Loss'], row['Rec_Prob_12month_Stable']]
    class_names = ['Growth', 'Loss', 'Stable']
    dominant_idx = int(np.argmax(probs))
    return class_names[dominant_idx], cwci

def probabilistic_discounting(row):
    # Use max hinder as discount factor
    hinders = [get_hinder_value(row['Image_Condition']), get_hinder_value(row['Spatial_Context']), get_hinder_value(row['Spectral_Capacity'])]
    discount = max(hinders) / 100.0
    raw = np.array([row['Rec_Prob_12month_Growth'], row['Rec_Prob_12month_Loss'], row['Rec_Prob_12month_Stable']], dtype=float)
    if raw.sum() == 0:
        raw = np.array([1,0,0], dtype=float) # fallback
    raw = raw / raw.sum()
    n = len(raw)
    discounted = (1 - discount) * raw + discount * (1.0 / n)
    return discounted

def main():
    # Always use the absolute path to the input CSV file
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, '1.rawOutputs')
    trimmed_dir = os.path.join(base_dir, '2.trimmedOutputs')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(trimmed_dir, exist_ok=True)
    input_csv = os.path.abspath(os.path.join(base_dir, '..', 'referenceIntObservationsFull.csv'))
    df = pd.read_csv(input_csv)
    # Prepare for all three methods
    # 1. Binary Heuristic Thresholding
    binary_rows = []
    for _, row in df.iterrows():
        label, is_confident = binary_heuristic_thresholding(row)
        binary_rows.append({'id': row['tracker'], 'label': label, 'is_confident': is_confident})
    binary_df = pd.DataFrame(binary_rows)
    binary_standardizer = ProbStandardizer(['Growth', 'Loss', 'Stable'], id_col='id')
    binary_out = binary_standardizer.from_binary_confidence(binary_df, label_col='label', is_confident_col='is_confident')
    binary_out.to_csv(os.path.join(raw_dir, '1.hfff_binary.csv'), index=False)

    # 2. Continuous Weighted Confidence Index
    conf_rows = []
    for _, row in df.iterrows():
        label, conf = continuous_weighted_confidence(row)
        conf_rows.append({'id': row['tracker'], 'label': label, 'confidence': conf})
    conf_df = pd.DataFrame(conf_rows)
    conf_standardizer = ProbStandardizer(['Growth', 'Loss', 'Stable'], id_col='id')
    conf_out = conf_standardizer.from_confidence(conf_df, label_col='label', confidence_col='confidence')
    conf_out.to_csv(os.path.join(raw_dir, '2.hfff_cwci.csv'), index=False)

    # 3. Probabilistic Discounting
    discount_rows = []
    for _, row in df.iterrows():
        discounted = probabilistic_discounting(row)
        discount_rows.append({'id': row['tracker'], 'Growth': discounted[0], 'Loss': discounted[1], 'Stable': discounted[2]})
    discount_df = pd.DataFrame(discount_rows)
    discount_standardizer = ProbStandardizer(['Growth', 'Loss', 'Stable'], id_col='id')
    discount_out = discount_standardizer.from_counts(discount_df, id_col='id')
    discount_out.to_csv(os.path.join(raw_dir, '3.hfff_discount.csv'), index=False)

    # Composite file
    composite = pd.DataFrame({'id': df['tracker']})
    composite = composite.merge(binary_out, on='id', suffixes=('', '_binary'))
    composite = composite.merge(conf_out, on='id', suffixes=('', '_cwci'))
    composite = composite.merge(discount_out, on='id', suffixes=('', '_discount'))
    composite.to_csv(os.path.join(raw_dir, '4.hfff_composite.csv'), index=False)

if __name__ == '__main__':
    main()
