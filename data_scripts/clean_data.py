"""
SmogShield – Data Cleaning Pipeline
====================================
Cleans all CSV datasets from the downloaded zip and writes
cleaned outputs to data/raw/, data/processed/, and data/archive_unused/
(mirroring the same sub-directory layout that already exists in the repo).

Run from the project root:
    python data_scripts/clean_data.py

Cleaning operations per file
------------------------------
1. openaq_us_post_25135_hourly.csv  (raw/ground_truth)
   - Parse datetime columns to UTC-aware timestamps
   - Drop columns that are 100 % null (coordinates, parameter.displayName,
     summary.sd) — structurally empty in this extract
   - Remove rows where the primary measurement value is NaN
     (sensor read-failures; 8 997 rows, ~19 %)
   - Validate that pm25 values are non-negative
   - Remove exact duplicate rows (none found, but handled defensively)

2. city_info.csv  (raw/validation)
   - No missing values; validate expected column types
   - Defensive duplicate check

3. data_dictionary.csv  (raw/validation)
   - unit column has 6 NaN → replace with empty string "—"
   - Defensive duplicate check

4. modeled_cams_lahore_daily_2022_2026.csv  (raw/validation)
   - Parse date column
   - First 3 rows are all-NaN data rows (CAMS model spin-up gap):
     dropped with a threshold rule (>= 90 % of numeric cols missing)
   - Remaining NaN cells in us_aqi / european_aqi (1 row each) filled
     via linear interpolation on the time axis

5. modeled_cams_lahore_hourly_2023_2025.csv  (raw/validation)
   - Parse time column
   - Validate no negatives in PM2.5 / PM10
   - No missing values; defensive duplicate check

6. station_aqi_weather_lahore_daily_2019_2023.csv  (raw/validation)
   - Parse date column (format DD-MM-YY)
   - 221 missing aqi_pm2.5 values (~13 %): filled with forward-fill
     then backward-fill (time-series appropriate; avoids distributional
     shift from global mean imputation)
   - Validate temperature plausibility (Fahrenheit range 20–140 °F)
   - Validate min_pressure_in==0.0 outliers (physically impossible):
     replace zeros in pressure with NaN then interpolate

7. open_meteo_era5_lahore_daily_2019_2026.csv  (raw/weather)
   - File has a 3-row Open-Meteo metadata header; skip on read
   - Rename columns: strip units from column names for clean column IDs
   - Parse time column
   - 1 missing row (each numeric column): interpolate linearly
   - Validate no negative precipitation

8. lahore_us_post_pm25_daily_2019_2025.csv  (processed)
   - Parse date_local column
   - 276 days where pm25_mean_ugm3 is NaN (usable_day already False):
     retain rows but log them (they are already flagged usable_day=False)
   - Remove exact duplicates

9. lahore_us_post_pm25_hourly_2019_2025.csv  (processed)
   - Parse timestamp columns
   - 8 997 missing pm25_ugm3 values (same sensor read-failures as raw):
     drop these rows to keep only valid measurements
   - Remove exact duplicates

10. lahore_complete_data_july_to_dec_2024.csv  (archive_unused)
    - Parse datetime column
    - Validate no missing values (confirmed clean)
    - Validate non-negative pollutant values

11. pakistan_air_quality_annual_summary.csv  (archive_unused)
    - 30 NaN in pakistan_national_avg and pakistan_global_rank
      (data only available from a certain year onwards):
      fill national_avg from who_ratio_annual * 5.0 (WHO limit proxy)
      and leave global_rank as NaN (cannot be inferred)
    - Defensive duplicate check

12. weather.csv  (archive_unused)
    - Parse date column (timezone-aware string)
    - Validate no missing values (confirmed clean)
    - Defensive duplicate check
"""

import os
import re
import logging
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_FOLDER = os.path.join(BASE_DIR, "zip_extracted")   # extracted zip contents
OUT_ROOT   = os.path.join(BASE_DIR, "data")            # repo data/ folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smogshield.clean")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_csv(rel_path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV from the extracted zip folder."""
    path = os.path.join(ZIP_FOLDER, rel_path)
    df = pd.read_csv(path, **kwargs)
    log.info("Read  %-60s  shape=%s", rel_path, df.shape)
    return df


def save_csv(df: pd.DataFrame, rel_path: str) -> None:
    """Write cleaned DataFrame to data/ folder, creating dirs as needed."""
    out_path = os.path.join(OUT_ROOT, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved %-60s  shape=%s", rel_path, df.shape)


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are 100 % NaN."""
    before = list(df.columns)
    df = df.dropna(axis=1, how="all")
    dropped = [c for c in before if c not in df.columns]
    if dropped:
        log.info("  Dropped 100%%-null columns: %s", dropped)
    return df


def drop_mostly_null_rows(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Drop rows where >= threshold fraction of numeric columns are NaN."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df
    null_frac = df[num_cols].isnull().mean(axis=1)
    mask = null_frac >= threshold
    n_dropped = mask.sum()
    if n_dropped:
        log.info("  Dropped %d rows with >= %.0f%% numeric NaN", n_dropped, threshold * 100)
    return df[~mask].copy()


def report_missing(df: pd.DataFrame, label: str) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        log.info("  [%s] Missing values:\n%s", label, missing.to_string())
    else:
        log.info("  [%s] No missing values.", label)


def remove_duplicates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n_dup = df.duplicated().sum()
    if n_dup:
        log.info("  [%s] Removing %d duplicate rows.", label, n_dup)
        df = df.drop_duplicates().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 1. openaq_us_post_25135_hourly.csv  (raw/ground_truth)
# ---------------------------------------------------------------------------
def clean_openaq_hourly():
    label = "openaq_us_post_25135_hourly"
    df = read_csv("raw/ground_truth/openaq_us_post_25135_hourly.csv")

    original_shape = df.shape

    # --- Drop columns that are structurally all-null in this extract ---
    df = drop_all_null_columns(df)

    # --- Parse datetime columns ---
    datetime_cols = [
        "date_utc", "date_local",
        "period.datetimeTo.utc", "period.datetimeTo.local",
        "coverage.datetimeFrom.utc", "coverage.datetimeFrom.local",
        "coverage.datetimeTo.utc", "coverage.datetimeTo.local",
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # --- Remove rows where the primary pm25 value is missing ---
    n_before = len(df)
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    log.info("  [%s] Dropped %d rows with missing `value` (sensor read failures).",
             label, n_before - len(df))

    # --- Validate non-negative PM2.5 ---
    neg_mask = df["value"] < 0
    if neg_mask.any():
        log.warning("  [%s] %d negative PM2.5 values clamped to 0.", label, neg_mask.sum())
        df.loc[neg_mask, "value"] = 0.0

    # --- Remove exact duplicates ---
    df = remove_duplicates(df, label)

    report_missing(df, label)
    log.info("  [%s] Rows: %d → %d | Cols: %d → %d",
             label, original_shape[0], len(df), original_shape[1], len(df.columns))

    save_csv(df, "raw/ground_truth/openaq_us_post_25135_hourly.csv")


# ---------------------------------------------------------------------------
# 2. city_info.csv  (raw/validation)
# ---------------------------------------------------------------------------
def clean_city_info():
    label = "city_info"
    df = read_csv("raw/validation/city_info.csv")

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "raw/validation/city_info.csv")


# ---------------------------------------------------------------------------
# 3. data_dictionary.csv  (raw/validation)
# ---------------------------------------------------------------------------
def clean_data_dictionary():
    label = "data_dictionary"
    df = read_csv("raw/validation/data_dictionary.csv")

    # 6 NaN in `unit` column → replace with placeholder dash
    if "unit" in df.columns:
        n_filled = df["unit"].isnull().sum()
        df["unit"] = df["unit"].fillna("—")
        log.info("  [%s] Filled %d NaN in `unit` with '—'.", label, n_filled)

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "raw/validation/data_dictionary.csv")


# ---------------------------------------------------------------------------
# 4. modeled_cams_lahore_daily_2022_2026.csv  (raw/validation)
# ---------------------------------------------------------------------------
def clean_cams_daily():
    label = "modeled_cams_lahore_daily"
    df = read_csv("raw/validation/modeled_cams_lahore_daily_2022_2026.csv")

    # --- Parse date ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- Drop model spin-up rows (>=90% numeric cols null) ---
    df = drop_mostly_null_rows(df, threshold=0.90)

    # --- Interpolate remaining NaN (us_aqi, european_aqi) ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    before_na = df[num_cols].isnull().sum().sum()
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    after_na = df[num_cols].isnull().sum().sum()
    log.info("  [%s] Interpolated %d numeric NaN cells.", label, before_na - after_na)

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "raw/validation/modeled_cams_lahore_daily_2022_2026.csv")


# ---------------------------------------------------------------------------
# 5. modeled_cams_lahore_hourly_2023_2025.csv  (raw/validation)
# ---------------------------------------------------------------------------
def clean_cams_hourly():
    label = "modeled_cams_lahore_hourly"
    df = read_csv("raw/validation/modeled_cams_lahore_hourly_2023_2025.csv")

    # --- Parse time ---
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # --- Validate non-negative pollutants ---
    for col in ["pm2_5", "pm10"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                log.warning("  [%s] %d negative values in '%s' clamped to 0.", label, neg, col)
                df.loc[df[col] < 0, col] = 0.0

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "raw/validation/modeled_cams_lahore_hourly_2023_2025.csv")


# ---------------------------------------------------------------------------
# 6. station_aqi_weather_lahore_daily_2019_2023.csv  (raw/validation)
# ---------------------------------------------------------------------------
def clean_station_aqi_weather():
    label = "station_aqi_weather_lahore_daily"
    df = read_csv("raw/validation/station_aqi_weather_lahore_daily_2019_2023.csv")

    # --- Parse date  (format: DD-MM-YY) ---
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%y", errors="coerce")

    # Sort chronologically before time-series imputation
    df = df.sort_values("date").reset_index(drop=True)

    # --- Forward-fill then back-fill aqi_pm2.5 ---
    col_aqi = "aqi_pm2.5"
    n_missing = df[col_aqi].isnull().sum()
    df[col_aqi] = df[col_aqi].ffill().bfill()
    log.info("  [%s] Forward/back-filled %d NaN in '%s'.", label, n_missing, col_aqi)

    # --- Fix zero-pressure outliers (physically impossible; likely missing coded as 0) ---
    pressure_cols = ["max_pressure_in", "avg_pressure_in", "min_pressure_in"]
    for pcol in pressure_cols:
        if pcol in df.columns:
            zero_mask = df[pcol] == 0.0
            if zero_mask.any():
                log.info("  [%s] Replacing %d zero(s) in '%s' with NaN then interpolating.",
                         label, zero_mask.sum(), pcol)
                df.loc[zero_mask, pcol] = np.nan
                df[pcol] = df[pcol].interpolate(method="linear", limit_direction="both")

    # --- Fix temperature outliers (Fahrenheit; Lahore context: 20–140 °F) ---
    # min_temp_f == 0: physically impossible; treat as missing sensor read (53 rows)
    # max_temp_f == 196: single typo (likely 96 °F — consistent with avg/min that day)
    for tcol in ["max_temp_f", "avg_temp_f", "min_temp_f"]:
        if tcol in df.columns:
            # Zero minimum: replace with NaN then interpolate
            zero_mask = (df[tcol] == 0)
            if zero_mask.any():
                log.info(
                    "  [%s] Replacing %d zero(s) in '%s' (impossible min-temp) "
                    "with NaN then interpolating.",
                    label, zero_mask.sum(), tcol,
                )
                df.loc[zero_mask, tcol] = np.nan
                df[tcol] = df[tcol].interpolate(method="linear", limit_direction="both")
            # High-end outliers (> 140 °F): replace with NaN then interpolate
            high_mask = df[tcol] > 140
            if high_mask.any():
                log.info(
                    "  [%s] Replacing %d value(s) > 140 °F in '%s' "
                    "(likely data-entry error) with NaN then interpolating.",
                    label, high_mask.sum(), tcol,
                )
                df.loc[high_mask, tcol] = np.nan
                df[tcol] = df[tcol].interpolate(method="linear", limit_direction="both")

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "raw/validation/station_aqi_weather_lahore_daily_2019_2023.csv")


# ---------------------------------------------------------------------------
# 7. open_meteo_era5_lahore_daily_2019_2026.csv  (raw/weather)
# ---------------------------------------------------------------------------
def clean_era5_weather():
    label = "open_meteo_era5_lahore_daily"

    # File has a 3-row Open-Meteo metadata header; skip it
    df = read_csv("raw/weather/open_meteo_era5_lahore_daily_2019_2026.csv", skiprows=3)

    # --- Rename columns: strip unit strings in parentheses ---
    rename_map = {}
    for col in df.columns:
        clean_name = re.sub(r"\s*\(.*?\)", "", col).strip()          # remove (unit)
        clean_name = clean_name.replace(" ", "_").lower()             # snake_case
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")
        rename_map[col] = clean_name
    df.rename(columns=rename_map, inplace=True)
    log.info("  [%s] Renamed columns to snake_case: %s", label, list(df.columns))

    # --- Parse time column ---
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # --- Interpolate the 1 missing row (all numeric cols) ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    before_na = df[num_cols].isnull().sum().sum()
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    after_na = df[num_cols].isnull().sum().sum()
    log.info("  [%s] Interpolated %d numeric NaN cell(s).", label, before_na - after_na)

    # --- Validate non-negative precipitation ---
    for pcol in ["precipitation_sum", "rain_sum"]:
        if pcol in df.columns:
            neg = (df[pcol] < 0).sum()
            if neg:
                log.warning("  [%s] %d negative values in '%s' clamped to 0.", label, neg, pcol)
                df.loc[df[pcol] < 0, pcol] = 0.0

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "raw/weather/open_meteo_era5_lahore_daily_2019_2026.csv")


# ---------------------------------------------------------------------------
# 8. lahore_us_post_pm25_daily_2019_2025.csv  (processed)
# ---------------------------------------------------------------------------
def clean_pm25_daily():
    label = "lahore_us_post_pm25_daily"
    df = read_csv("processed/lahore_us_post_pm25_daily_2019_2025.csv")

    # --- Parse date ---
    df["date_local"] = pd.to_datetime(df["date_local"], errors="coerce")

    # Sort chronologically
    df = df.sort_values("date_local").reset_index(drop=True)

    # --- Log but retain NaN pm25_mean rows (already flagged usable_day=False) ---
    n_missing_pm25 = df["pm25_mean_ugm3"].isnull().sum()
    if n_missing_pm25:
        log.info(
            "  [%s] %d rows with NaN pm25_mean_ugm3 are retained "
            "(usable_day=False already set). These correspond to days "
            "with insufficient sensor coverage.",
            label, n_missing_pm25
        )

    # --- Validate non-negative pm25 ---
    if "pm25_mean_ugm3" in df.columns:
        neg = (df["pm25_mean_ugm3"].dropna() < 0).sum()
        if neg:
            log.warning("  [%s] %d negative pm25_mean_ugm3 clamped to 0.", label, neg)
            df.loc[df["pm25_mean_ugm3"] < 0, "pm25_mean_ugm3"] = 0.0

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "processed/lahore_us_post_pm25_daily_2019_2025.csv")


# ---------------------------------------------------------------------------
# 9. lahore_us_post_pm25_hourly_2019_2025.csv  (processed)
# ---------------------------------------------------------------------------
def clean_pm25_hourly():
    label = "lahore_us_post_pm25_hourly"
    df = read_csv("processed/lahore_us_post_pm25_hourly_2019_2025.csv")

    # --- Parse timestamps ---
    df["timestamp_utc_period_start"] = pd.to_datetime(
        df["timestamp_utc_period_start"], utc=True, errors="coerce"
    )
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], errors="coerce"
    )

    # --- Drop rows with missing pm25_ugm3 (sensor read-failures) ---
    n_before = len(df)
    df = df.dropna(subset=["pm25_ugm3"]).reset_index(drop=True)
    log.info("  [%s] Dropped %d rows with missing pm25_ugm3 (sensor read-failures).",
             label, n_before - len(df))

    # --- Validate non-negative PM2.5 ---
    neg = (df["pm25_ugm3"] < 0).sum()
    if neg:
        log.warning("  [%s] %d negative pm25_ugm3 clamped to 0.", label, neg)
        df.loc[df["pm25_ugm3"] < 0, "pm25_ugm3"] = 0.0

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "processed/lahore_us_post_pm25_hourly_2019_2025.csv")


# ---------------------------------------------------------------------------
# 10. lahore_complete_data_july_to_dec_2024.csv  (archive_unused)
# ---------------------------------------------------------------------------
def clean_archive_complete():
    label = "lahore_complete_data_july_to_dec_2024"
    df = read_csv("archive_unused/lahore_complete_data_july_to_dec_2024.csv")

    # --- Parse datetime (mixed formats: M/D/YYYY H:MM  and  DD/MM/YYYY HH:MM:SS) ---
    # Format 1 (1 704 rows): M/D/YYYY H:MM   e.g. "1/7/2024 0:00"
    # Format 2 (2 664 rows): DD/MM/YYYY HH:MM:SS e.g. "13/07/2024 00:00:00"
    parsed_f1 = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    parsed_f2 = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["datetime"] = parsed_f1.fillna(parsed_f2)
    still_null = df["datetime"].isnull().sum()
    if still_null:
        log.warning("  [%s] %d datetime values could not be parsed and remain NaT.",
                    label, still_null)
    else:
        log.info("  [%s] All %d datetime values parsed successfully "
                 "(two-format mixed parsing: M/D/YYYY H:MM + DD/MM/YYYY HH:MM:SS).",
                 label, len(df))

    # --- Validate non-negative pollutants ---
    pollutant_cols = [
        "components_co", "components_no", "components_no2",
        "components_o3", "components_so2", "components_pm2_5",
        "components_pm10", "components_nh3",
    ]
    for pcol in pollutant_cols:
        if pcol in df.columns:
            neg = (df[pcol] < 0).sum()
            if neg:
                log.warning("  [%s] %d negative values in '%s' clamped to 0.", label, neg, pcol)
                df.loc[df[pcol] < 0, pcol] = 0.0

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "archive_unused/lahore_complete_data_july_to_dec_2024.csv")


# ---------------------------------------------------------------------------
# 11. pakistan_air_quality_annual_summary.csv  (archive_unused)
# ---------------------------------------------------------------------------
def clean_archive_annual():
    label = "pakistan_air_quality_annual_summary"
    df = read_csv("archive_unused/pakistan_air_quality_annual_summary.csv")

    # --- pakistan_national_avg: only available from certain years ---
    # Rows where national_avg is NaN are pre-2019 data (no national network).
    # We cannot reliably infer the national average from other columns,
    # so we fill with the overall mean of available values.
    if "pakistan_national_avg" in df.columns:
        n_missing = df["pakistan_national_avg"].isnull().sum()
        if n_missing:
            col_mean = df["pakistan_national_avg"].mean()
            df["pakistan_national_avg"] = df["pakistan_national_avg"].fillna(col_mean)
            log.info(
                "  [%s] Filled %d NaN in 'pakistan_national_avg' with "
                "column mean (%.2f). NOTE: These are pre-network years; "
                "treat with caution.",
                label, n_missing, col_mean,
            )

    # --- pakistan_global_rank: cannot be inferred; leave as NaN ---
    if "pakistan_global_rank" in df.columns:
        n_missing = df["pakistan_global_rank"].isnull().sum()
        if n_missing:
            log.info(
                "  [%s] %d NaN in 'pakistan_global_rank' retained "
                "(rank cannot be imputed reliably from other columns).",
                label, n_missing,
            )

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "archive_unused/pakistan_air_quality_annual_summary.csv")


# ---------------------------------------------------------------------------
# 12. weather.csv  (archive_unused)
# ---------------------------------------------------------------------------
def clean_archive_weather():
    label = "archive_weather"
    df = read_csv("archive_unused/weather.csv")

    # --- Parse date (timezone-aware ISO string) ---
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # --- Validate non-negative rain ---
    if "rain" in df.columns:
        neg = (df["rain"] < 0).sum()
        if neg:
            log.warning("  [%s] %d negative rain values clamped to 0.", label, neg)
            df.loc[df["rain"] < 0, "rain"] = 0.0

    df = remove_duplicates(df, label)
    report_missing(df, label)

    save_csv(df, "archive_unused/weather.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("SmogShield Data Cleaning Pipeline — started")
    log.info("=" * 70)

    clean_openaq_hourly()
    clean_city_info()
    clean_data_dictionary()
    clean_cams_daily()
    clean_cams_hourly()
    clean_station_aqi_weather()
    clean_era5_weather()
    clean_pm25_daily()
    clean_pm25_hourly()
    clean_archive_complete()
    clean_archive_annual()
    clean_archive_weather()

    log.info("=" * 70)
    log.info("All files cleaned and saved to data/")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
