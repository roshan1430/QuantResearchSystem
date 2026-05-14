# Datasets

This project uses NASA and CERN as the research framing so the platform feels closer to live scientific operations instead of generic IoT telemetry.

## Files
- `sample_timeseries.csv`: engineered multi-sensor telemetry sample shaped for local training, inspired by cryogenic and instrument-monitoring behavior.
- `nasa_apod_sample.json`: cached example of a NASA-style external event payload.
- `cern_live_sample.json`: cached example of a CERN-style structured event payload.
- `build_research_dataset.py`: converts actual NASA POWER hourly data into a training-ready CSV for this platform.
- `research_pipeline.py`: reusable transformation logic for actual NASA research data.

## Research positioning
- NASA feeds are useful for proving public live-source ingestion and external event persistence.
- CERN-style feeds are useful for demonstrating experimental infrastructure, detector, or accelerator monitoring concepts.
- The internal time-series sample remains synthetic, but it is now documented as a stand-in for research telemetry rather than arbitrary dummy data.

## Column notes
- `timestamp`: UTC event time used for sorting and walk-forward validation.
- `sensor_id`: sensor identity for grouped analysis.
- `temperature_k`, `pressure_atm`, `vibration_mms`: raw measurements.
- `rolling_mean_16`, `rolling_std_16`, `z_score_16`, `momentum_8`, `lag_1`, `volatility_16`, `ema_12`: engineered features expected by the ML engine.
- `target_temperature`: short-horizon supervised target for regression and forecasting experiments.

## Recommended usage
- Use `sample_timeseries.csv` with `POST /train` for quick local forecasting validation.
- Use `nasa_apod_sample.json` and a configured CERN JSON endpoint to demo the external-source flow.
- Use `python datasets/build_research_dataset.py --site cern --start 20250101 --end 20250107` to build a stronger training dataset from official NASA POWER hourly data over the CERN area.
- Use the notebook scaffold in `notebooks/` to inspect drift, feature quality, and per-sensor behavior before training.

## Important honesty note
- Actual model training can now be strengthened with real NASA POWER time-series data.
- CERN support in this repo is still strongest as an external research-event context source unless you provide a timestamped CERN JSON feed that matches the platform's schema.
