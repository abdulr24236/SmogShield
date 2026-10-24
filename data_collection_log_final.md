# Data Collection Log — SmogShield Lahore

Evidence of data gathering for **Deliverable 1, Phase 2 (Collection)**.
Every file below is a public source. Raw files are stored on the shared Google Drive
folder and are **not** committed to GitHub (see `.gitignore`).

- Shared raw-data folder: <https://drive.google.com/drive/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y>
- Download script: [`data_scripts/01_download_data.py`](../data_scripts/01_download_data.py)
- Re-run with: `python data_scripts/01_download_data.py` (OpenAQ needs `OPENAQ_API_KEY` in the environment)

---

## 1. Source summary

| # | Local file | Role | Source | Access method | Coverage | Rows | License |
|---|---|---|---|---|---|---|---|
| 1 | `raw/ground_truth/openaq_us_post_25135_hourly.csv` | Main target — hourly PM2.5 | **OpenAQ v3**, sensor `25135`, location `8664` (US Diplomatic Post, Lahore) | REST API, 90-day chunks | 2019-05-22 → 2025-03-04 | 46,465 | Provider-specific (US State Dept. via OpenAQ) |
| 2 | `raw/weather/open_meteo_era5_lahore_daily_2019_2026.csv` | Main weather features | **Open-Meteo Historical Weather API**, model `era5` | REST API | 2019-06-01 → 2026-10-08 | 2,687 | CC BY 4.0 |
| 3 | `raw/validation/station_aqi_weather_lahore_daily_2019_2023.csv` | Cross-check + secondary weather | **GitHub** `taimoorbarlas15/Lahore_Air_Quality` → `lahore_aqi_2019_to_2023.csv` | Raw file download | 2019-06-01 → 2023-11-30 | 1,644 | Not stated ⚠️ |
| 4 | `raw/validation/modeled_cams_lahore_hourly_2023_2025.csv` | Modeled hourly comparison | Delivered from **Kaggle** `aadbutt/dataset`; regenerable from **Open-Meteo Air-Quality API** (`cams_global`) | API / manual Kaggle download | 2023-01-01 00:00 → 2025-05-01 23:00 | 20,448 | Kaggle: Unknown ⚠️ / API: CC BY 4.0 |
| 5 | `raw/validation/modeled_cams_lahore_daily_2022_2026.csv` | Modeled daily comparison (optional) | **Zenodo** DOI [10.5281/zenodo.18673725](https://doi.org/10.5281/zenodo.18673725) (also Kaggle `nitirajkulkarni/lahore-pk-1172451`) | Zip download | 2022-08-01 → 2026-02-18 | 1,298 | **CC0-1.0** |
| 6 | `raw/validation/city_info.csv` | Metadata | Same Zenodo/Kaggle dataset as #5 | Zip download | — | 1 | **CC0-1.0** |
| 7 | `raw/validation/data_dictionary.csv` | Metadata | Same Zenodo/Kaggle dataset as #5 | Zip download | — | 21 | **CC0-1.0** |

> Note: source #5 is a single dataset published by **Nitiraj Kulkarni & Jagadish Tawade**
> on both Zenodo (`lahore-pk-1172451.zip`) and Kaggle (`nitirajkulkarni/lahore-pk-1172451`).
> The file we hold is **byte-identical** to `air_quality_historical.csv` inside that zip
> (237,253 bytes), so Zenodo and Kaggle are the same data.

---

## 2. Per-source detail

### 2.1 OpenAQ — sensor 25135 (main target)

- **Endpoint:** `GET https://api.openaq.org/v3/sensors/25135/measurements/hourly`
- **Auth:** header `X-API-Key`, read from the `OPENAQ_API_KEY` environment variable.
- **Paging:** `limit=1000`, `page=1..totalPages`; 90-day date chunks.
- **Location:** OpenAQ location id `8664`, "US Diplomatic Post: Lahore".
- **Delivered columns (33):** `value`, `coordinates`, `flagInfo.hasFlags`,
  `parameter.*`, `units`, `period.*`, `date_utc`, `date_local`, `summary.*`,
  `coverage.*`, `sensor_id`.
- **Time base:** `date_utc` is the **start** of each hourly period (UTC);
  `date_local` is the same instant in PKT (UTC+5).
- **Quality flags:** `flagInfo.hasFlags` is `True` on 9,596 rows; 1,018 of those
  still carry a value. Whether to exclude flagged values remains an **open decision**.

### 2.2 Open-Meteo ERA5 — daily weather

- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive`
- **Parameters:** `latitude=31.558`, `longitude=74.35071`, `models=era5`,
  `timezone=Asia/Karachi`, `start_date=2019-06-01`, `end_date=2026-10-07` in the script.
  The delivered file was requested through 2026-10-08 (that last row is empty), so a fresh run of the
  script returns one row fewer (2,686) than the delivered file (2,687).
- **Daily variables:** temperature (mean/max/min), relative humidity (mean),
  dew point (mean), surface pressure (mean), wind speed (mean/max), dominant
  wind direction, precipitation sum, rain sum, shortwave radiation sum, cloud
  cover (mean).
- **File format:** 3 metadata lines (latitude, longitude, elevation, timezone)
  followed by the header row — read with `skiprows=3`.
- **Last row:** 2026-10-08 is empty because of the ERA5 reanalysis delay, so
  **2,686 usable days**.

### 2.3 GitHub station file

- **URL:** `https://raw.githubusercontent.com/taimoorbarlas15/Lahore_Air_Quality/main/lahore_aqi_2019_to_2023.csv`
- **Evidence of match:** identical header and identical first rows to our local
  `station_aqi_weather_lahore_daily_2019_2023.csv` (e.g. `01-06-19,222,115,101,88,…`).
- **Format issues to clean:** dates are `dd-mm-yy`; temperature in °F; wind in
  mph; pressure in inches; zeros in min-temperature / min-dew-point /
  min-humidity / min-pressure are invalid.

### 2.4 CAMS hourly (modeled comparison)

- **Delivered file:** Kaggle dataset [`aadbutt/dataset`](https://www.kaggle.com/datasets/aadbutt/dataset)
  — "Lahore Weather & AQI - Hourly (2023-2025)", by `$aad Butt`.
  The description lists the same 12 columns and the same coverage
  (1 Jan 2023 → 2 May 2025). Kaggle reports licence **"Unknown"**.
- **Reproducible alternative (used by the script):** Open-Meteo Air-Quality API
  (`cams_global`) for pollutants plus the weather archive API for the
  meteorological columns, merged on the same hourly timestamps.
- **Columns:** `time`, `pm2_5`, `pm10`, `ozone`, `nitrogen_dioxide`,
  `sulphur_dioxide`, `carbon_monoxide`, `temperature_2m`,
  `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `pressure_msl`.
- **Caveat:** modeled, **not measured**; must never be treated as ground truth.

### 2.5 Zenodo / Kaggle `lahore-pk-1172451`

- **DOI:** `10.5281/zenodo.18673725` (published 2026-02-17).
- **Authors:** Nitiraj Kulkarni, Jagadish Tawade.
- **Zip contents:** `air_quality_historical.csv`, `city_info.csv`,
  `data_dictionary.csv`, `dataset-metadata.json`, `README.md`.
- **Mapping used:** `air_quality_historical.csv` →
  `modeled_cams_lahore_daily_2022_2026.csv`; the other two files keep their names.
- **Values:** generated from the Open-Meteo Air-Quality service (`cams_global`)
  and GeoNames city metadata.
- **Licence:** **CC0-1.0**.

---

## 3. File fingerprints (SHA-256)

Computed on the delivered raw files. These let anyone verify that a re-download
produced the same file.

| File | Bytes | Data rows | SHA-256 |
|---|---:|---:|---|
| `openaq_us_post_25135_hourly.csv` | 14,391,821 | 46,465 | `efe92e940639013aeb190497953d98dfe7027ed42caa6a0763c7048fac99370e` |
| `open_meteo_era5_lahore_daily_2019_2026.csv` | 190,860 | 2,687 | `8e61f72c5b03dd7b24902de5fb3434a865791535b507fb7f5dcd97241d39ea62` |
| `station_aqi_weather_lahore_daily_2019_2023.csv` | 113,340 | 1,644 | `7bf1f53eaea1602eb87e8c1ab97985d62b4cb496953a4147af8c28d7697754f0` |
| `modeled_cams_lahore_hourly_2023_2025.csv` | 1,375,293 | 20,448 | `860a804f9a96a0fb414f50d07d0470bff158429d089aec0cf96032e4a8b9bd2d` |
| `modeled_cams_lahore_daily_2022_2026.csv` | 237,253 | 1,298 | `ce1327775720bae72986a04edeab0cfdce0f778642be499570bdf8e6d152b0fb` |
| `city_info.csv` | 152 | 1 | `cdefaf01c862d270613165a67012174ea844031a0ad080d7f86d258ff535ebf5` |
| `data_dictionary.csv` | 1,274 | 21 | `208057b0417c49429e7be95d8a9c256837893af6f369ebfbff8e97bf0db12373` |

> Row counts exclude headers and the 3 metadata lines of the ERA5 file.
> Re-downloads may differ slightly for live APIs (new rows, revised values);
> the hashes pin the exact snapshot used in this deliverable.

---

## 4. Reproducing the download

```bash
# all sources (OpenAQ is skipped unless the key is set)
python data_scripts/01_download_data.py

# only the sources that need no key
python data_scripts/01_download_data.py --only weather --only cams-hourly --only zenodo --only station

# OpenAQ only
export OPENAQ_API_KEY=...          # bash
$env:OPENAQ_API_KEY="..."          # PowerShell
python data_scripts/01_download_data.py --only openaq
```

The script appends a dated section with row counts, byte sizes and SHA-256
hashes to the bottom of this file.

---

## 5. Reproduction test results

The script was tested locally against the delivered files:

| Source | Result |
|---|---|
| GitHub station | **Byte-identical** — SHA-256 matches the delivered file exactly |
| Zenodo zip (daily CAMS, `city_info`, `data_dictionary`) | **Byte-identical** — all three SHA-256 hashes match |
| Open-Meteo ERA5 daily weather | Same structure (3 metadata lines + header); a fresh request reports elevation **219 m** (delivered file: 220 m) and slightly revised values, so byte equality is not expected |
| Open-Meteo CAMS hourly | **20,448 rows** — same row count and columns as the delivered Kaggle snapshot; values may differ slightly in the last digits |
| OpenAQ | Not run locally (no key); the flattening reproduces the 33 delivered columns |

> Live APIs are revised over time. The SHA-256 hashes in section 3 pin the exact
> snapshot used in this deliverable; a re-download is expected to be equivalent,
> not necessarily byte-identical, for the ERA5 and CAMS sources.

---

## 6. Open confirmations

| Item | Status |
|---|---|
| OpenAQ provider licence/credit wording | Resolved: provider-specific; credit OpenAQ and StateAir Lahore (US Department of State network). See `DATA_SOURCES` §1.1 and OpenAQ's Terms of Use. |
| Open-Meteo / ERA5 attribution wording (CC BY 4.0) | Resolved: credit "Weather data by Open-Meteo.com", link the CC BY 4.0 licence, and note column/unit standardisation. See `DATA_SOURCES` §1.2. |
| `aadbutt` Kaggle licence | Unknown — attribute the dataset |
| `taimoorbarlas15` GitHub licence | No licence stated — attribute the repository |
| Whether to exclude OpenAQ flagged values | Open decision (test both) |
| AQI breakpoints | **US EPA 2012** (decided in Phase 1) |
| Shared Google Drive link | Added above |
