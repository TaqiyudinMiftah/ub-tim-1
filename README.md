# PasarPulse Multimodal Prototype

Reproducible data pipeline, baseline modeling, evaluation, and deployment scaffold for multimodal Indonesian food-price forecasting.

The automated experiment lives under `pasarpulse/` and is executed by GitHub Actions. It downloads public price data, weather data, vegetation-index data, builds leakage-safe time features, trains price-only and multimodal baselines, and packages the processed datasets, models, predictions, plots, and metrics as a workflow artifact.
