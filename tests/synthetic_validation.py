import numpy as np
import pandas as pd

from acc_assessment.gue import GUE
from acc_assessment.mcem import MCEM


CLASS_NAMES = ["A", "B"]
POPULATION_SIZE = 100_000
SAMPLE_SIZE = 500


def _interval_from_standard_error(estimate, standard_error):
    margin = 1.96 * standard_error
    return estimate - margin, estimate + margin


def _stratified_sample(dataframe, strata_col, n_samples, rng):
    strata_counts = dataframe[strata_col].value_counts().sort_index()
    strata_props = strata_counts / strata_counts.sum()

    allocations = (strata_props * n_samples).round().astype(int).to_dict()
    for stratum in allocations:
        allocations[stratum] = max(1, allocations[stratum])

    diff = n_samples - sum(allocations.values())
    ordered = list(strata_counts.sort_values(ascending=False).index)
    idx = 0
    while diff != 0:
        stratum = ordered[idx % len(ordered)]
        if diff > 0 and allocations[stratum] < int(strata_counts[stratum]):
            allocations[stratum] += 1
            diff -= 1
        elif diff < 0 and allocations[stratum] > 1:
            allocations[stratum] -= 1
            diff += 1
        idx += 1

    sampled = []
    for stratum, n_h in allocations.items():
        stratum_rows = dataframe[dataframe[strata_col] == stratum]
        sampled_idx = rng.choice(stratum_rows.index.values, size=n_h, replace=False)
        sampled.append(stratum_rows.loc[sampled_idx])

    return pd.concat(sampled).sort_values("id").reset_index(drop=True)


def build_synthetic_population(random_state=2026):
    rng = np.random.default_rng(random_state)

    true_class = rng.choice(CLASS_NAMES, size=POPULATION_SIZE, p=[0.01, 0.99])

    map_probs = np.zeros((POPULATION_SIZE, 2), dtype=float)
    map_probs[true_class == "A"] = [1.0, 0.0]
    map_probs[true_class == "B"] = [0.0, 1.0]

    map_noise_mask = (true_class == "A") & (rng.random(POPULATION_SIZE) < 0.20)
    map_probs[map_noise_mask] = [0.3, 0.7]

    ref_probs = np.zeros((POPULATION_SIZE, 2), dtype=float)
    ref_probs[true_class == "A"] = [1.0, 0.0]
    ref_probs[true_class == "B"] = [0.0, 1.0]

    edge_mask = rng.random(POPULATION_SIZE) < 0.02
    ref_probs[edge_mask] = [0.5, 0.5]

    strata = np.where(map_probs[:, 0] >= map_probs[:, 1], "A_stratum", "B_stratum")

    population = pd.DataFrame(
        {
            "id": np.arange(POPULATION_SIZE),
            "strata": strata,
            "A_map": map_probs[:, 0],
            "B_map": map_probs[:, 1],
            "A_ref": ref_probs[:, 0],
            "B_ref": ref_probs[:, 1],
        }
    )

    true_area_a = float(np.sum(ref_probs[:, 0]))
    true_overall_accuracy = float(np.mean(np.sum(map_probs * ref_probs, axis=1)))

    return population, true_area_a, true_overall_accuracy


def main():
    rng = np.random.default_rng(2026)
    population, true_area_a, true_overall_accuracy = build_synthetic_population(
        random_state=2026
    )

    strata_population = population["strata"].value_counts().to_dict()
    sample = _stratified_sample(population, "strata", SAMPLE_SIZE, rng)

    map_table = sample[["A_map", "B_map", "strata", "id"]].rename(
        columns={"A_map": "A", "B_map": "B"}
    )
    ref_table = sample[["A_ref", "B_ref", "strata", "id"]].rename(
        columns={"A_ref": "A", "B_ref": "B"}
    )

    gue = GUE(
        map_data=map_table,
        ref_data=ref_table,
        strata_col="strata",
        id_col="id",
        strata_population=strata_population,
    )
    mcem = MCEM(
        map_data=map_table,
        ref_data=ref_table,
        strata_col="strata",
        id_col="id",
        strata_population=strata_population,
        n_simulations=4000,
        random_state=2026,
    )

    gue_area_estimate, gue_area_se = gue.area("A", reference=True)
    mcem_area_mean, mcem_area_ci = mcem.area("A", reference=True)
    gue_area_ci = _interval_from_standard_error(gue_area_estimate, gue_area_se)

    gue_oa_estimate, gue_oa_se = gue.overall_accuracy()
    mcem_oa_mean, mcem_oa_ci = mcem.overall_accuracy()
    gue_oa_ci = _interval_from_standard_error(gue_oa_estimate, gue_oa_se)

    area_in_gue_ci = gue_area_ci[0] <= true_area_a <= gue_area_ci[1]
    area_in_mcem_ci = mcem_area_ci[0] <= true_area_a <= mcem_area_ci[1]

    oa_in_gue_ci = gue_oa_ci[0] <= true_overall_accuracy <= gue_oa_ci[1]
    oa_in_mcem_ci = mcem_oa_ci[0] <= true_overall_accuracy <= mcem_oa_ci[1]

    print(
        f"True Area: {true_area_a:.2f} | "
        f"GUE Estimate: {gue_area_estimate:.2f} | "
        f"MCEM Mean: {mcem_area_mean:.2f}"
    )
    print(f"GUE Area 95% CI: ({gue_area_ci[0]:.2f}, {gue_area_ci[1]:.2f})")
    print(f"MCEM Area 95% CI: ({mcem_area_ci[0]:.2f}, {mcem_area_ci[1]:.2f})")
    print(f"True Area within GUE 95% CI: {area_in_gue_ci}")
    print(f"True Area within MCEM 95% CI: {area_in_mcem_ci}")

    print(
        f"True Accuracy: {true_overall_accuracy:.6f} | "
        f"GUE Estimate: {gue_oa_estimate:.6f} | "
        f"MCEM Mean: {mcem_oa_mean:.6f}"
    )
    print(f"GUE Accuracy 95% CI: ({gue_oa_ci[0]:.6f}, {gue_oa_ci[1]:.6f})")
    print(f"MCEM Accuracy 95% CI: ({mcem_oa_ci[0]:.6f}, {mcem_oa_ci[1]:.6f})")
    print(f"True Accuracy within GUE 95% CI: {oa_in_gue_ci}")
    print(f"True Accuracy within MCEM 95% CI: {oa_in_mcem_ci}")


if __name__ == "__main__":
    main()