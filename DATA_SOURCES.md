# Data Sources

Documentation of every dataset used, considered, or rejected in SmogShield. Last updated: 2026-10-24.

**Conventions**
- Raw files in `data/raw/` are never edited. Cleaned files in `data/processed/` are produced by scripts in `data_scripts/`.
- Raw data is not stored in this repository (see `.gitignore`). Shared copy: <https://drive.google.com/drive/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y>. Download dates, hashes and the exact requests are in `reports/data_collection_log.md`.
- "Probable" or "inferred" means the origin was deduced from the file contents and was not documented by whoever supplied the file. These items must be confirmed before the final report.
- Pakistan Standard Time (PKT) is UTC+5. "Local day" below means the PKT calendar day.

## Summary

| File | Folder | Role | Coverage | Resolution | Status |
|---|---|---|---|---|---|
| `openaq_us_post_25135_hourly.csv` | `raw/ground_truth` | **Main target (PM2.5)** | 2019-05-22 to 2025-03-04 | Hourly | Used |
| `open_meteo_era5_lahore_daily_2019_2026.csv` | `raw/weather` | **Main weather features** | 2019-06-01 to 2026-10-07 | Daily | Used |
| `station_aqi_weather_lahore_daily_2019_2023.csv` | `raw/validation` | Cross-check, secondary weather experiment | 2019-06-01 to 2023-11-30 | Daily | Used |
| `modeled_cams_lahore_hourly_2023_2025.csv` | `raw/validation` | Modeled comparison, hourly pollutants | 2023-01-01 to 2025-05-01 | Hourly | Used |
| `modeled_cams_lahore_daily_2022_2026.csv` (+ `city_info.csv`, `data_dictionary.csv`) | `raw/validation` | Modeled comparison, optional extension | 2022-08-01 to 2026-02-18 | Daily | Optional |
| `pm25_hourly.csv` | `processed` | Cleaned hourly PM2.5 | 2019-05-22 to 2025-03-04 | Hourly | Derived (script 02) |
| `pm25_daily_clean.csv` | `processed` | Cleaned daily PM2.5, AQI and category | 2019-05-23 to 2025-03-04 | Daily | Derived (scripts 02, 03) |
| `station_daily.csv` | `processed` | Cleaned station AQI and weather | 2019-06-01 to 2023-11-30 | Daily | Derived (script 04) |
| `lahore_complete_data_july_to_dec_2024.csv` | `archive_unused` | Not used | 2024-07-01 to 2024-12-31 | Hourly | Archived |
| `weather.csv` | `archive_unused` | Not used | 2014-01-01 to 2024-01-01 | Daily | Archived |
| `pakistan_air_quality_annual_summary.csv` | `archive_unused` | Not used | 2015 to 2025 | Annual | Archived |

---

## 1. Files used

### 1.1 `openaq_us_post_25135_hourly.csv` (main target)
- **What:** hourly PM2.5 (µg/m³) at the US Diplomatic Post monitor in Lahore.
- **Source:** OpenAQ API v3, sensor `25135`, location "US Diplomatic Post: Lahore" (OpenAQ location id `8664`). Data provider is the US Department of State network via OpenAQ (as listed by OpenAQ).
- **Access method:** endpoint `/v3/sensors/25135/measurements/hourly`, downloaded in 90-day chunks. Download date: **15 October 2026** (timestamp of the delivered snapshot).
- **Coverage:** 2019-05-22 to 2025-03-04 (UTC). 46,465 rows, 33 columns, no duplicate timestamps.
- **Timestamps:** `date_utc` = start of the hour, UTC.
- **Quality:**
  - 37,468 hours have a value (about 74% of the 50,703 hours in the span). 4,238 hours have no row, and 8,997 rows have an empty `value`.
  - 9,596 rows carry quality flags. 1,018 of them still have a value. Whether to exclude flagged values is an open decision and should be tested.
  - Maximum 974 µg/m³, 648 hours above 500, 24 exact zeros. Extremes are plausible during smog episodes but should be checked.
  - Longest gaps: 26 Oct 2022 to 3 Jan 2023 (70 days), 21 Oct to 8 Dec 2021 (49 days), Apr to May 2024 (50 days), Feb to Mar 2022 (39 days).
- **Consistency checks:** matches an earlier extract of the same sensor exactly on 7,951 overlapping hours; daily AQI built from this sensor correlates 0.99 with the station AQI in 1.3 over 1,153 overlapping days (mean absolute difference 9.3 AQI points).
- **License / attribution:** Provider-specific. OpenAQ lists the provider as "StateAir Lahore" (US Department of State air-quality network) and requires attribution to the original source and to OpenAQ; the delivered snapshot does not include a separate licence field. We credit OpenAQ and the provider and do not redistribute the raw file. See OpenAQ's Terms of Use: <https://docs.openaq.org/about/terms>.

### 1.2 `open_meteo_era5_lahore_daily_2019_2026.csv` (main weather)
- **What:** daily weather for Lahore from ERA5 reanalysis.
- **Source:** Open-Meteo Historical Weather API (`archive-api.open-meteo.com`), `models=era5`, `timezone=Asia/Karachi`, requested point 31.558 N, 74.351 E. The file header shows the grid cell used: 31.5 N, 74.25 E, elevation 220 m.
- **Variables (daily):** temperature (mean, max, min), relative humidity (mean), dew point (mean), surface pressure (mean), wind speed (mean, max), dominant wind direction, precipitation sum, rain sum, shortwave radiation sum, cloud cover (mean). Units are in the column headers.
- **Coverage:** 2019-06-01 to 2026-10-08. 2,687 rows, no missing dates. The last row (2026-10-08) is empty because of the reanalysis delay, so 2,686 usable days. The first 3 lines of the file are a metadata header (read with `skiprows=3`).
- **Quality:** agrees closely with station temperature (correlation 0.99) and dew point (0.99), reasonably with humidity (0.94), and less with wind (0.64) and rain (0.4 to 0.7 depending on the reference).
- **License / attribution:** Open-Meteo data are licensed under CC BY 4.0. Attribution: "Weather data by Open-Meteo.com" (<https://open-meteo.com/>), with a link to the licence (<https://creativecommons.org/licenses/by/4.0/>). ERA5 data: generated using Copernicus Climate Change Service information (2026); neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. We retrieved the data through the Historical Weather API and standardised column names and units; no values were otherwise changed.

### 1.3 `station_aqi_weather_lahore_daily_2019_2023.csv` (cross-check)
- **What:** daily Lahore PM2.5 AQI plus weather columns (temperature in °F, wind in mph, pressure in inches).
- **Source:** GitHub repository `taimoorbarlas15/Lahore_Air_Quality`, file `lahore_aqi_2019_to_2023.csv` (`raw.githubusercontent.com`). The downloaded file is byte-identical to ours (SHA-256 in the collection log). The repository's own sources are not documented. Its AQI behaves like the US Diplomatic Post data (correlation 0.99 with 1.1 converted to AQI), so it is probably from the same monitor (inferred).
- **Coverage:** 2019-06-01 to 2023-11-30, 1,644 rows, no missing dates, dates formatted `dd-mm-yy`.
- **Quality:** AQI is missing on 221 days (13.4%), including a 47-day gap from 22 Oct to 7 Dec 2021. Zeros in the minimum temperature, dew point, humidity and pressure columns are invalid and should be treated as missing. Seven average pressure values are implausibly low.
- **Cleaning applied (script 04):** impossible weather values set to empty (323), units converted to match Open-Meteo, AQI left unfilled, weather averages forward-filled for at most 3 days.
- **License:** none stated in the repository, so we attribute it and do not redistribute it.

### 1.4 `modeled_cams_lahore_hourly_2023_2025.csv` (modeled comparison)
- **What:** hourly modeled pollutants and weather for Lahore.
- **Source:** Kaggle dataset `aadbutt/dataset` ("Lahore Weather & AQI - Hourly (2023-2025)"), per the collection log. The field names match Open-Meteo's air-quality and weather APIs, which serve CAMS global atmospheric-composition data at about 0.4° (roughly 45 km) from August 2022, so it can be regenerated from them. The file's timestamps are **UTC**; the cleaning script converts them to Pakistan time.
- **Coverage:** 2023-01-01 00:00 to 2025-05-01 23:00, 20,448 rows, no missing hours. No rain column, and no location in the file (matches other Lahore files).
- **Quality:** modeled, not measured. Against the ground-truth daily means it reads about half as high (ratio 0.53 over 604 overlapping days) and under-reads smog peaks, although its day-to-day pattern tracks well (correlation 0.83 for Jul to Dec 2024). `aqi_category`-style columns are derived from PM2.5 and must not be used as inputs.
- **License:** Kaggle lists it as "Unknown"; the regenerable API route is CC BY 4.0. We attribute it and do not redistribute it.

### 1.5 `modeled_cams_lahore_daily_2022_2026.csv`, `city_info.csv`, `data_dictionary.csv` (optional)
- **What:** daily modeled pollutants (PM2.5, PM10, CO, NO₂, SO₂, ozone, aerosol optical depth, dust, UV index, US AQI, European AQI) for Lahore. `city_info.csv` gives the city coordinates (31.558 N, 74.351 E) and `data_dictionary.csv` lists column units.
- **Source:** Zenodo DOI 10.5281/zenodo.18673725 (Nitiraj Kulkarni and Jagadish Tawade, published 2026-02-17; also on Kaggle as `nitirajkulkarni/lahore-pk-1172451`), licence **CC0-1.0**. The three files match the archive's hashes (collection log). Its daily PM2.5 agrees with 1.4 (correlation 0.985).
- **Coverage:** 2022-08-01 to 2026-02-18, 1,298 rows, no missing dates. The first 3 rows are empty and one `us_aqi` value is missing.
- **Quality:** same limitations as 1.4. Against the monitor over 632 shared days it averaged 0.54 of the monitor's PM2.5 (median daily ratio 0.61). Against the station AQI over 394 overlapping days it correlates 0.67, averages 27 AQI points lower, and reads about 207 to 270 on days when the station reported 447 to 491. The daily `us_aqi` is an average of hourly values, not the official daily AQI.
- **Use:** only for a calibration or comparison experiment, never merged directly with the ground truth.

## 2. Derived files (reproducible)

All derived files are created by scripts in `data_scripts/` and are shared on Google Drive (`cleaned_v1`), not in this repository.

| Output | Script | Input | Key counts |
|---|---|---|---|
| `pm25_hourly.csv` | `02_clean_pm25_simple.py` | `ground_truth/openaq_us_post_25135_hourly.csv` | 46,465 rows loaded, 0 duplicates, 0 negatives, 8,997 empty readings; full span 50,703 hours with 37,468 valid; 24 exact zeros; 0 outliers flagged (3 x IQR on log scale, kept in data) |
| `pm25_daily.csv` | `02_clean_pm25_simple.py` | the hourly series above | 2,113 calendar days, 1,506 usable (at least 18 valid hours) |
| `pm25_daily_clean.csv` | `03_pm25_aqi_and_eda.py` | `pm25_daily.csv` | `pm25_clean` filled only on usable days, plus `aqi` and `aqi_category` (US EPA PM2.5 breakpoints, 2012 version). Classes: Good 9, Moderate 126, Unhealthy for Sensitive 269, Unhealthy 647, Very Unhealthy 256, Hazardous 139 + 60 |
| `calendar_daily.csv` | `08_build_calendar.py` | none (built from the `holidays` and `hijridate` packages) | 2,802 days, 2019-05-01 to 2026-12-31; weekend = Saturday and Sunday; Islamic dates are calculated and can be one day off |
| `features_daily.csv` | `09_build_features.py` | `pm25_daily_clean.csv` + raw ERA5 weather | 2,113 rows, 63 columns; 1,179 rows with a next-day target and lags 1 to 3 |
| `station_daily.csv` | `04_clean_station_simple.py` | `validation/station_aqi_weather_lahore_daily_2019_2023.csv` | 1,644 rows; 323 impossible values set to empty; 221 AQI days left empty (only 2 recoverable from OpenAQ); 27 days above AQI 400 |

Rules applied everywhere: raw files are never edited; the target (PM2.5 or AQI) is never filled; extremes are flagged, not deleted; a day is usable only with at least 18 valid hours; missing hours stay as empty rows.

Cross-check: AQI computed from the cleaned US Post series agrees with the station AQI on 1,153 overlapping days (correlation 0.99, mean absolute difference 9.3 points, ours 4.9 points higher on average).

Open decisions to write in the report: (1) the 1,018 quality-flagged hours that still have a PM2.5 value are kept, with a sensitivity test planned; (2) the 24 exact-zero hours are not yet inspected; (3) the AQI breakpoint version (2012 used here; EPA revised its PM2.5 breakpoints in 2024).

## 3. Archived (kept for reference, not used)

| File | Why archived |
|---|---|
| `lahore_complete_data_july_to_dec_2024.csv` | Hourly, Jul to Dec 2024, with pollutant fields in the style of the OpenWeather Air Pollution API (probable) and weather including rain. Source undocumented, and the date column mixes two formats (`d/m/yyyy h:mm` and `dd/mm/yyyy hh:mm:ss`). Its PM2.5 averages 183 µg/m³ against 148 in the ground truth for the same days (correlation 0.69), overshooting in September and December. |
| `weather.csv` | Daily weather for 15 cities, 2014 to 2024. Source undocumented. Replaced by 1.2, which covers more variables and dates. Timestamps are at 19:00 UTC (the next local day). |
| `pakistan_air_quality_annual_summary.csv` | Annual city summaries, 2015 to 2025. Across all 110 city-years the peak month is almost always about 1.85 times the annual mean, which suggests formula-generated values. Source unknown. |

## 4. Reviewed and rejected (not in the repository)

| File or source | Reason |
|---|---|
| `traffic_congestion_dataset.csv` | Synthetic: negative vehicle counts, flat counts across the day, congestion as a step function, random segment IDs, no link to Lahore, only Jan to Feb 2024. |
| `final-dataset.csv` | Hourly Mar to Jun 2024 with only AQI, PM2.5, temperature and humidity. Unknown source and location, smoothed values, poor agreement with other Lahore data (hourly PM2.5 correlation 0.26), and timestamps about 5 to 6 hours off. |
| `pakistan_air_quality_final_clean_v2.csv` | 10 cities, about 3 months of one winter, probably modeled. Redundant with 1.4. |
| `dataset.xlsx` | Daily "Conc." plus weather, Jan 2022 to Dec 2024. The pollutant measured and the source are unknown, and it under-reads on the worst smog days. |
| `lahore_v3_historical.csv` | OpenAQ export with every date missing and the wrong query parameters. Superseded by 1.1. |
| OpenAQ export for location 1894641 | 5,000 rows covering 14 days of one low-cost sensor. A full download for sensor `7466365` could extend coverage past March 2025 and is optional. |
| 2019 to 2020 four-city AQI spreadsheet (opendata.com.pk) | One year only, superseded by 1.1 and 1.3. |
| Punjab EPD / PMD station sheets (from `Saba-Gul/Forecasting-Air-Quality-Index`) | Mixed and swapped date formats, placeholder cells, copied rows between sheets, large disagreement with the ground truth, no license. Ends April 2020. |
| Code repositories `muhammadadilnaeem/Lahore-Air-Quality-Index-Predicter` and `Hanan-Nawaz/Pakistan_AQI_Analysis` | Used only to identify the US Diplomatic Post sensor. Their code is not reused. The second repository contains no data. |

## 5. Traffic and calendar data
No open, time-aligned Lahore traffic dataset was found. Traffic is represented by **calendar proxies**: weekday, weekend (Saturday and Sunday), public holidays, Ramadan and Eid. `08_build_calendar.py` builds the full table from the `holidays` and `hijridate` packages. Islamic dates are calculated, so they can be one day off from the day Pakistan announced after moon sighting. School terms are not included. `09_build_features.py` currently uses its own simplified holiday list (fixed dates plus Eid).

## 6. Pipeline scripts (`data_scripts/`)

Run from the main project folder, in this order:

| Script | Purpose |
|---|---|
| `01_download_data.py` | Downloads the API and public sources, writes hashes to `reports/data_collection_log.md` (OpenAQ key from `OPENAQ_API_KEY`) |
| `02_clean_pm25_simple.py` | Cleans the OpenAQ hourly file, builds the hourly and daily series |
| `03_pm25_aqi_and_eda.py` | Model-ready PM2.5, AQI and category, four EDA charts |
| `04_clean_station_simple.py` | Cleans the station AQI and weather file |
| `05`-`07` (Ayub) | Clean the ERA5 weather file and the two CAMS files |
| `08_build_calendar.py` | Builds the calendar table |
| `09_build_features.py` | Builds the feature table and two more EDA charts |

