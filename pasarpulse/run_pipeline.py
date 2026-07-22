from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TARGET_COMMODITIES = {"Bawang Merah", "Cabai Rawit"}


def log(message: str) -> None:
    print(message, flush=True)


def clean_location(value: object) -> str:
    text = str(value).strip()
    replacements = {
        "Kabupaten ": "Kab. ",
        "Kota ": "",
        "Kab ": "Kab. ",
        "  ": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    aliases = {
        "Jakarta": "DKI Jakarta",
        "Jakarta Raya": "DKI Jakarta",
        "Kab. Cirebon ": "Kab. Cirebon",
        "Kab. Tasikmalaya ": "Kab. Tasikmalaya",
    }
    return aliases.get(text, text)


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matched {pattern} under {root}")
    return matches[-1]


def copy_source_inputs(source_root: Path, out_raw: Path) -> dict[str, str]:
    out_raw.mkdir(parents=True, exist_ok=True)
    sources = {
        "price": source_root / "cleaned_pihps_data" / "cleaned_combined.csv",
        "weather": source_root / "weather_pihps_historical.csv",
        "production": source_root / "bps-jakarta-data" / "jawa_barat_food_production_2024.csv",
        "spatial": find_one(source_root, "supply_chain_spatial_fast/spatial_features_*.csv"),
        "facility_summary": find_one(source_root, "supply_chain_spatial_fast/location_facility_summary_*.csv"),
        "province_geojson": source_root / "GeoJSON" / "Indonesia_provinces.geojson",
    }
    manifest = {}
    for name, path in sources.items():
        if not path.exists():
            log(f"WARNING: optional source missing: {path}")
            continue
        dest = out_raw / path.name
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            dest = dest.with_suffix(".csv.gz")
            df.to_csv(dest, index=False, compression="gzip")
        else:
            shutil.copy2(path, dest)
        manifest[name] = str(dest.relative_to(out_raw.parent.parent))
    return manifest


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    expected = {"date", "commodity_name", "location_name", "price"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Price dataset missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["commodity_name"] = df["commodity_name"].astype(str).str.strip()
    df["location"] = df["location_name"].map(clean_location)
    df = df.dropna(subset=["date", "price"])
    df = df[(df["price"] > 1000) & (df["price"] < 400_000)]

    shallot = df[df["commodity_name"].str.fullmatch("Bawang Merah", case=False, na=False)].copy()
    shallot["commodity"] = "Bawang Merah"

    rawit = df[df["commodity_name"].isin(["Cabai Rawit Merah", "Cabai Rawit Hijau"])].copy()
    rawit["commodity"] = "Cabai Rawit"

    result = pd.concat([shallot, rawit], ignore_index=True)
    result = (
        result.groupby(["date", "location", "commodity"], as_index=False)
        .agg(price=("price", "median"), price_observations=("price", "size"))
        .sort_values(["location", "commodity", "date"])
    )
    return result


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["location"] = df["location_name"].map(clean_location)
    numeric = [
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "precipitation_mm",
        "rain_mm",
        "precipitation_hours",
        "windspeed_max_kmh",
        "windgusts_max_kmh",
        "latitude",
        "longitude",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["date", "location"] + [c for c in numeric if c in df.columns]
    return df[keep].dropna(subset=["date", "location"]).drop_duplicates(["date", "location"])


def load_production(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["location"] = df["nama_kabupaten_kota"].map(clean_location)
    frames = []
    mapping = {
        "Bawang Merah": "produksi_bawang_merah_ton",
        "Cabai Rawit": "produksi_cabai_rawit_ton",
    }
    for commodity, col in mapping.items():
        if col not in df.columns:
            continue
        part = df[["location", col]].copy()
        part["commodity"] = commodity
        part["production_ton"] = pd.to_numeric(part[col], errors="coerce")
        frames.append(part[["location", "commodity", "production_ton"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["location", "commodity", "production_ton"])


def load_spatial(spatial_path: Path, summary_path: Path) -> pd.DataFrame:
    spatial = pd.read_csv(spatial_path)
    summary = pd.read_csv(summary_path)
    spatial["location"] = spatial["location"].map(clean_location)
    summary["location"] = summary["location"].map(clean_location)

    def map_commodity(value: object) -> str:
        text = str(value).strip().lower()
        if "bawang merah" in text:
            return "Bawang Merah"
        if "cabai" in text:
            return "Cabai Rawit"
        return str(value).strip()

    spatial["commodity"] = spatial["commodity"].map(map_commodity)
    summary["commodity"] = summary["commodity"].map(map_commodity)
    merged = spatial.merge(summary, on=["location", "commodity"], how="outer", suffixes=("", "_summary"))
    numeric_cols = [c for c in merged.columns if c not in {"location", "commodity"}]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged[merged["commodity"].isin(TARGET_COMMODITIES)].drop_duplicates(["location", "commodity"])


def modis_dates(latitude: float, longitude: float) -> list[dict]:
    url = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/dates"
    response = requests.get(url, params={"latitude": latitude, "longitude": longitude}, timeout=90)
    response.raise_for_status()
    return response.json()


def fetch_modis_year(latitude: float, longitude: float, start_modis: str, end_modis: str) -> list[dict]:
    url = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "startDate": start_modis,
        "endDate": end_modis,
        "kmAboveBelow": 1,
        "kmLeftRight": 1,
        "band": "250m_16_days_NDVI",
    }
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    payload = response.json()
    scale = float(payload.get("scale", 0.0001) or 0.0001)
    rows = []
    for item in payload.get("subset", []):
        values = np.asarray(item.get("data", []), dtype=float)
        valid = values[(values >= -2000) & (values <= 10000)]
        rows.append(
            {
                "date": pd.to_datetime(item.get("calendar_date"), errors="coerce"),
                "ndvi_mean": float(np.nanmean(valid) * scale) if len(valid) else np.nan,
                "ndvi_std": float(np.nanstd(valid) * scale) if len(valid) else np.nan,
                "ndvi_valid_fraction": float(len(valid) / len(values)) if len(values) else 0.0,
            }
        )
    return rows


def download_ndvi(weather: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if not cached.empty:
            return cached
    locations = (
        weather.groupby("location", as_index=False)
        .agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        .dropna()
    )
    records = []
    for i, row in locations.iterrows():
        location = row["location"]
        lat, lon = float(row["latitude"]), float(row["longitude"])
        log(f"MODIS NDVI {i + 1}/{len(locations)}: {location}")
        try:
            dates = modis_dates(lat, lon)
            available = [
                d for d in dates
                if start <= pd.Timestamp(d["calendar_date"]) <= end
            ]
            by_year: dict[int, list[dict]] = {}
            for d in available:
                by_year.setdefault(pd.Timestamp(d["calendar_date"]).year, []).append(d)
            for year, items in sorted(by_year.items()):
                items = sorted(items, key=lambda x: x["calendar_date"])
                try:
                    yearly = fetch_modis_year(lat, lon, items[0]["modis_date"], items[-1]["modis_date"])
                    for obs in yearly:
                        obs["location"] = location
                        obs["latitude"] = lat
                        obs["longitude"] = lon
                        records.append(obs)
                except Exception as exc:
                    log(f"WARNING: MODIS subset failed for {location} {year}: {exc}")
                time.sleep(0.2)
        except Exception as exc:
            log(f"WARNING: MODIS dates failed for {location}: {exc}")
    result = pd.DataFrame(records)
    if not result.empty:
        result = result.dropna(subset=["date"]).drop_duplicates(["date", "location"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    return result


def asof_ndvi(base: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
    if ndvi.empty:
        base = base.copy()
        for col in ["ndvi_mean", "ndvi_std", "ndvi_valid_fraction", "ndvi_age_days"]:
            base[col] = np.nan
        return base
    parts = []
    for location, group in base.groupby("location", sort=False):
        left = group.sort_values("date")
        right = ndvi[ndvi["location"] == location].sort_values("date")
        if right.empty:
            left = left.copy()
            for col in ["ndvi_mean", "ndvi_std", "ndvi_valid_fraction", "ndvi_date", "ndvi_age_days"]:
                left[col] = np.nan
        else:
            right = right[["date", "ndvi_mean", "ndvi_std", "ndvi_valid_fraction"]].rename(columns={"date": "ndvi_date"})
            left = pd.merge_asof(left, right, left_on="date", right_on="ndvi_date", direction="backward")
            left["ndvi_age_days"] = (left["date"] - left["ndvi_date"]).dt.days
        parts.append(left)
    return pd.concat(parts, ignore_index=True)


def add_neighbor_price(df: pd.DataFrame, weather: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    coords = weather.groupby("location")[["latitude", "longitude"]].median().dropna()
    locations = list(coords.index)
    if len(locations) < 2:
        df["neighbor_price_mean"] = np.nan
        return df
    arr = coords[["latitude", "longitude"]].to_numpy(float)
    dist = np.sqrt(((arr[:, None, :] - arr[None, :, :]) ** 2).sum(axis=2))
    neighbor_map = {}
    for i, loc in enumerate(locations):
        order = np.argsort(dist[i])
        neighbor_map[loc] = [locations[j] for j in order if j != i][:k]

    pivot = df.pivot_table(index=["date", "commodity"], columns="location", values="price", aggfunc="median")
    neighbor_series = []
    for loc in df["location"].unique():
        neighbors = [n for n in neighbor_map.get(loc, []) if n in pivot.columns]
        if not neighbors:
            continue
        s = pivot[neighbors].mean(axis=1).rename("neighbor_price_mean").reset_index()
        s["location"] = loc
        neighbor_series.append(s)
    if not neighbor_series:
        df["neighbor_price_mean"] = np.nan
        return df
    neighbors = pd.concat(neighbor_series, ignore_index=True)
    return df.merge(neighbors, on=["date", "commodity", "location"], how="left")


def build_master(prices: pd.DataFrame, weather: pd.DataFrame, production: pd.DataFrame, spatial: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
    valid_locations = set(weather["location"].unique())
    df = prices[prices["location"].isin(valid_locations)].copy()
    df = df.merge(weather, on=["date", "location"], how="left")
    df = asof_ndvi(df, ndvi)
    df = df.merge(production, on=["location", "commodity"], how="left")
    df = df.merge(spatial, on=["location", "commodity"], how="left")
    df = add_neighbor_price(df, weather)
    df = df.sort_values(["location", "commodity", "date"]).reset_index(drop=True)

    group = df.groupby(["location", "commodity"], group_keys=False)
    for lag in [1, 5, 10, 20]:
        df[f"price_lag_{lag}"] = group["price"].shift(lag)
    for window in [5, 10, 20]:
        df[f"price_roll_mean_{window}"] = group["price"].transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        df[f"price_roll_std_{window}"] = group["price"].transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())

    for source, prefix in [
        ("rain_mm", "rain"),
        ("temperature_mean_c", "temp"),
        ("windspeed_max_kmh", "wind"),
    ]:
        if source in df.columns:
            for window in [7, 30]:
                df[f"{prefix}_roll_{window}"] = group[source].transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365.25)
    df["target_price_5d"] = group["price"].shift(-5)
    df["target_return_5d"] = df["target_price_5d"] / df["price"] - 1
    df["shock_5d"] = (df["target_return_5d"] >= 0.10).astype(float)
    df.loc[df["target_price_5d"].isna(), "shock_5d"] = np.nan

    quality_cols = [
        "ndvi_mean", "production_ton", "nearest_production_km", "total_facilities",
        "rain_mm", "temperature_mean_c"
    ]
    for col in quality_cols:
        if col in df.columns:
            df[f"missing_{col}"] = df[col].isna().astype(int)
    return df


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / np.where(denom == 0, 1.0, denom)) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE_pct": smape(y_true, y_pred),
    }


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    result = {
        "PR_AUC": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ROC_AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Brier": float(brier_score_loss(y_true, prob)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
    }
    return result


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.75, 71)
    scores = [f1_score(y_true, prob >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def make_pipeline(numeric_features: list[str], categorical_features: list[str], classification: bool = False) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features),
        ],
        remainder="drop",
    )
    estimator = (
        HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=25,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
        if classification
        else HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            loss="absolute_error",
            random_state=RANDOM_STATE,
        )
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].dropna().unique())
    train_cut = pd.Timestamp(dates[int(len(dates) * 0.70)])
    val_cut = pd.Timestamp(dates[int(len(dates) * 0.85)])
    train = df[df["date"] <= train_cut].copy()
    val = df[(df["date"] > train_cut) & (df["date"] <= val_cut)].copy()
    test = df[df["date"] > val_cut].copy()
    return train, val, test


def plot_predictions(test: pd.DataFrame, output: Path) -> None:
    daily = test.groupby("date", as_index=False).agg(actual=("target_price_5d", "median"), predicted=("pred_multimodal", "median"), naive=("pred_naive", "median"))
    plt.figure(figsize=(11, 5))
    plt.plot(daily["date"], daily["actual"], label="Actual")
    plt.plot(daily["date"], daily["predicted"], label="Multimodal")
    plt.plot(daily["date"], daily["naive"], label="Naive", alpha=0.7)
    plt.title("PasarPulse 5-business-day price forecast: test-period median")
    plt.ylabel("Rp/kg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_model_comparison(metrics: pd.DataFrame, output: Path) -> None:
    reg = metrics[(metrics["task"] == "regression") & (metrics["metric"] == "MAE")].copy()
    plt.figure(figsize=(8, 4.5))
    plt.bar(reg["model"], reg["value"])
    plt.title("Test MAE by model")
    plt.ylabel("MAE (Rp/kg)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_report(output_dir: Path, manifest: dict, master: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, metrics: pd.DataFrame) -> None:
    reg = metrics[metrics["task"] == "regression"].pivot(index="model", columns="metric", values="value")
    cls = metrics[metrics["task"] == "classification"].pivot(index="model", columns="metric", values="value")
    report = [
        "# PasarPulse Multimodal Experiment Report",
        "",
        "## Dataset",
        f"- Master rows: {len(master):,}",
        f"- Locations: {master['location'].nunique()}",
        f"- Commodities: {', '.join(sorted(master['commodity'].dropna().unique()))}",
        f"- Date range: {master['date'].min().date()} to {master['date'].max().date()}",
        f"- Train/validation/test rows: {len(train):,} / {len(val):,} / {len(test):,}",
        "",
        "## Modalities",
        "1. Daily PIHPS retail prices.",
        "2. Daily Open-Meteo weather observations.",
        "3. MODIS MOD13Q1 16-day NDVI subsets from ORNL DAAC.",
        "4. BPS West Java production statistics.",
        "5. OpenStreetMap-derived supply-chain distance and facility features.",
        "6. Geographic-neighbor price signal.",
        "",
        "## Regression results",
        reg.round(3).to_markdown(),
        "",
        "## Price-shock classification results",
        cls.round(4).to_markdown() if not cls.empty else "No valid classification metrics.",
        "",
        "## Interpretation",
        "The key scientific comparison is price-only versus multimodal. An improvement on the held-out chronological test period supports the claim that weather, vegetation, production, spatial supply-chain, and neighbor-market signals add predictive value beyond price history. If the multimodal model does not improve, the result should be reported honestly and the modality alignment or feature quality must be revised.",
        "",
        "## Caveats",
        "- MODIS values are small point-centered subsets, not province-wide crop-specific observations.",
        "- Production is an annual structural feature and is unavailable for some demand-center locations.",
        "- OpenStreetMap facility completeness varies by place.",
        "- Results are a proof of concept for DKI Jakarta and West Java, not a national production model.",
        "",
        "## Packaged files",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("pasarpulse_output"))
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "data" / "raw"
    processed_dir = output_dir / "data" / "processed"
    model_dir = output_dir / "models"
    plot_dir = output_dir / "plots"
    for directory in [raw_dir, processed_dir, model_dir, plot_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    log("[1/8] Packaging source datasets")
    manifest = copy_source_inputs(source_root, raw_dir)

    price_path = source_root / "cleaned_pihps_data" / "cleaned_combined.csv"
    weather_path = source_root / "weather_pihps_historical.csv"
    production_path = source_root / "bps-jakarta-data" / "jawa_barat_food_production_2024.csv"
    spatial_path = find_one(source_root, "supply_chain_spatial_fast/spatial_features_*.csv")
    summary_path = find_one(source_root, "supply_chain_spatial_fast/location_facility_summary_*.csv")

    log("[2/8] Loading price, weather, production, and spatial modalities")
    prices = load_prices(price_path)
    weather = load_weather(weather_path)
    production = load_production(production_path)
    spatial = load_spatial(spatial_path, summary_path)

    start, end = prices["date"].min(), prices["date"].max()
    log("[3/8] Downloading MODIS NDVI")
    ndvi_path = raw_dir / "modis_ndvi_location_timeseries.csv"
    ndvi = download_ndvi(weather, start, end, ndvi_path)
    manifest["modis_ndvi"] = str(ndvi_path.relative_to(output_dir))

    log("[4/8] Building train-ready multimodal table")
    master = build_master(prices, weather, production, spatial, ndvi)
    master = master.dropna(subset=["target_price_5d", "price_lag_5"]).reset_index(drop=True)
    master.to_parquet(processed_dir / "master_multimodal.parquet", index=False)
    master.to_csv(processed_dir / "master_multimodal.csv.gz", index=False, compression="gzip")
    manifest["master_parquet"] = "data/processed/master_multimodal.parquet"
    manifest["master_csv_gz"] = "data/processed/master_multimodal.csv.gz"

    train, val, test = temporal_split(master)
    log(f"Temporal split: {len(train):,} train, {len(val):,} validation, {len(test):,} test")

    categorical = ["location", "commodity"]
    price_features = [
        "price", "price_lag_1", "price_lag_5", "price_lag_10", "price_lag_20",
        "price_roll_mean_5", "price_roll_mean_10", "price_roll_mean_20",
        "price_roll_std_5", "price_roll_std_10", "price_roll_std_20",
        "day_of_week", "month", "day_of_year_sin", "day_of_year_cos",
    ]
    multimodal_candidates = price_features + [
        "temperature_max_c", "temperature_min_c", "temperature_mean_c", "rain_mm",
        "precipitation_mm", "precipitation_hours", "windspeed_max_kmh", "windgusts_max_kmh",
        "rain_roll_7", "rain_roll_30", "temp_roll_7", "temp_roll_30", "wind_roll_7", "wind_roll_30",
        "ndvi_mean", "ndvi_std", "ndvi_valid_fraction", "ndvi_age_days",
        "production_ton", "nearest_production_km", "avg_production_distance_km",
        "nearest_distribution_km", "avg_distribution_distance_km", "nearest_retail_km",
        "total_facilities", "production_count", "distribution_count", "retail_count",
        "neighbor_price_mean",
    ]
    multimodal_features = [c for c in multimodal_candidates if c in master.columns]

    log("[5/8] Training regression baselines")
    price_model = make_pipeline(price_features, categorical, classification=False)
    multi_model = make_pipeline(multimodal_features, categorical, classification=False)
    price_model.fit(train[price_features + categorical], train["target_price_5d"])
    multi_model.fit(train[multimodal_features + categorical], train["target_price_5d"])

    test["pred_naive"] = test["price"]
    test["pred_price_only"] = price_model.predict(test[price_features + categorical])
    test["pred_multimodal"] = multi_model.predict(test[multimodal_features + categorical])

    metric_rows = []
    for model_name, pred_col in [
        ("Naive", "pred_naive"),
        ("PriceOnly", "pred_price_only"),
        ("Multimodal", "pred_multimodal"),
    ]:
        values = regression_metrics(test["target_price_5d"].to_numpy(), test[pred_col].to_numpy())
        for metric, value in values.items():
            metric_rows.append({"task": "regression", "model": model_name, "metric": metric, "value": value})

    log("[6/8] Training price-shock classifiers")
    train_cls = train.dropna(subset=["shock_5d"]).copy()
    val_cls = val.dropna(subset=["shock_5d"]).copy()
    test_cls = test.dropna(subset=["shock_5d"]).copy()
    price_cls = make_pipeline(price_features, categorical, classification=True)
    multi_cls = make_pipeline(multimodal_features, categorical, classification=True)
    price_cls.fit(train_cls[price_features + categorical], train_cls["shock_5d"].astype(int))
    multi_cls.fit(train_cls[multimodal_features + categorical], train_cls["shock_5d"].astype(int))

    val_prob_price = price_cls.predict_proba(val_cls[price_features + categorical])[:, 1]
    val_prob_multi = multi_cls.predict_proba(val_cls[multimodal_features + categorical])[:, 1]
    threshold_price = choose_threshold(val_cls["shock_5d"].astype(int).to_numpy(), val_prob_price)
    threshold_multi = choose_threshold(val_cls["shock_5d"].astype(int).to_numpy(), val_prob_multi)

    test_prob_price = price_cls.predict_proba(test_cls[price_features + categorical])[:, 1]
    test_prob_multi = multi_cls.predict_proba(test_cls[multimodal_features + categorical])[:, 1]
    test.loc[test_cls.index, "shock_prob_price_only"] = test_prob_price
    test.loc[test_cls.index, "shock_prob_multimodal"] = test_prob_multi
    for model_name, prob, threshold in [
        ("PriceOnly", test_prob_price, threshold_price),
        ("Multimodal", test_prob_multi, threshold_multi),
    ]:
        values = classification_metrics(test_cls["shock_5d"].astype(int).to_numpy(), prob, threshold)
        for metric, value in values.items():
            metric_rows.append({"task": "classification", "model": model_name, "metric": metric, "value": value})

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    test.to_csv(output_dir / "test_predictions.csv.gz", index=False, compression="gzip")

    joblib.dump(price_model, model_dir / "price_only_regressor.joblib")
    joblib.dump(multi_model, model_dir / "multimodal_regressor.joblib")
    joblib.dump(price_cls, model_dir / "price_only_shock_classifier.joblib")
    joblib.dump(multi_cls, model_dir / "multimodal_shock_classifier.joblib")
    (model_dir / "feature_schema.json").write_text(json.dumps({"categorical": categorical, "price_features": price_features, "multimodal_features": multimodal_features}, indent=2), encoding="utf-8")

    log("[7/8] Creating plots and report")
    plot_predictions(test, plot_dir / "test_forecast.png")
    plot_model_comparison(metrics, plot_dir / "model_comparison.png")
    manifest.update({
        "metrics": "metrics.csv",
        "test_predictions": "test_predictions.csv.gz",
        "models": "models/",
        "plots": "plots/",
    })
    write_report(output_dir, manifest, master, train, val, test, metrics)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("[8/8] Summary")
    print(metrics.pivot_table(index=["task", "model"], columns="metric", values="value").round(4).to_string())
    log(f"All outputs written to {output_dir}")


if __name__ == "__main__":
    main()
