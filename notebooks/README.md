# Notebooks

Use this folder for offline research that explains or validates the production pipeline without duplicating app logic.

## Included scaffold
- `research_walkthrough.py`: a script-friendly notebook starter for dataset inspection, summary stats, and forecasting prep.

## Suggested workflow
1. Generate a real dataset with `python datasets/build_research_dataset.py --site cern --start 20250101 --end 20250107` when network access is available.
2. Load either the generated NASA POWER dataset or `datasets/sample_timeseries.csv`.
3. Compare it with the cached NASA and CERN-style JSON examples in `datasets/`.
4. Inspect sensor-level drift and missing values.
5. Review feature-target correlation before changing the ML engine.
6. Promote stable feature logic back into `ml-engine/engine/services/` rather than keeping it notebook-only.
