# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Integrated probabilistic workflow now requires **four input files** instead of three:
  1. Map probabilities (`id` + class probabilities)
  2. Reference probabilities (`id` + class probabilities)
  3. Sample strata assignments (`id`, `strata`)
  4. Strata populations (`strata`, `population`)
- Added strict integrated input loading/validation path via `load_integrated_probability_inputs` and updated package exports.
- Updated README and notebook integrated examples/help text to reflect the new 4-file input contract.
- Updated notebook display labels for clarity:
  - `Generalized Unbiased Estimator ('gue_same')`
  - `Monte Carlo Error Matrix ('mcem_same')`

### Added
- New test fixture CSVs for split probabilistic input format:
  - `tests/map_data_probabilities_table.csv`
  - `tests/ref_data_probabilities_table.csv`
  - `tests/sample_strata_table.csv`
- Utility tests covering the strict 4-file loading behavior.
