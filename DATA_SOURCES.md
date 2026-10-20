# Data Sources

Documentation of every dataset used, considered, or rejected in SmogShield. Last updated: 2026-10-20.

**Conventions**
- Raw files in `data/raw/` are never edited. Cleaned files in `data/processed/` are produced by scripts in `data_scripts/`.
- Raw data is not stored in this repository (see `.gitignore`). Shared copy: **[Google Drive link: TO FILL]**.
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
| `lahore_us_post_pm25_hourly_2019_2025.csv` | `processed` | Cleaned hourly PM2.5 | 2019-05-22 to 2025-03-04 | Hourly | Derived |
| `lahore_us_post_pm25_daily_2019_2025.csv` | `processed` | Cleaned daily PM2.5 | 2019-05-22 to 2025-03-04 | Daily | Derived |
| `lahore_complete_data_july_to_dec_2024.csv` | `archive_unused` | Not used | 2024-07-01 to 2024-12-31 | Hourly | Archived |
| `weather.csv` | `archive_unused` | Not used | 2014-01-01 to 2024-01-01 | Daily | Archived |
| `pakistan_air_quality_annual_summary.csv` | `archive_unused` | Not used | 2015 to 2025 | Annual | Archived |

---

## 1. Files used

### 1.1 `openaq_us_post_25135_hourly.csv` (main target)
- **What:** hourly PM2.5 (µg/m³) at the US Diplomatic Post monitor in Lahore.
- **Source:** OpenAQ API v3, sensor `25135`, location "US Diplomatic Post: Lahore" (OpenAQ location id `8664`). Data provider is the US Department of State network via OpenAQ (as listed by OpenAQ).
- **Access method:** endpoint `/v3/sensors/25135/measurements/hourly`, downloaded in 90-day chunks. Download date: **TO FILL (October 2026)**.
- **Coverage:** 2019-05-22 to 2025-03-04 (UTC). 46,465 rows, 33 columns, no duplicate timestamps.
- **Timestamps:** `date_utc` = start of the hour, UTC.
- **Quality:**
  - 37,468 hours have a value (about 74% of the 50,703 hours in the span). 4,238 hours have no row, and 8,997 rows have an empty `value`.
  - 9,596 rows carry quality flags. 1,018 of them still have a value. Whether to exclude flagged values is an open decision and should be tested.
  - Maximum 974 µg/m³, 648 hours above 500, 24 exact zeros. Extremes are plausible during smog episodes but should be checked.
  - Longest gaps: 26 Oct 2022 to 3 Jan 2023 (70 days), 21 Oct to 8 Dec 2021 (49 days), Apr to May 2024 (50 days), Feb to Mar 2022 (39 days).
- **Consistency checks:** matches an earlier extract of the same sensor exactly on 7,951 overlapping hours; daily values converted to AQI correlate 0.99 with the station file in 1.3 (167 overlapping days).
- **License / attribution:** OpenAQ aggregates data under provider-specific licenses. **TO CONFIRM** the license for location 8664 and credit OpenAQ and the data provider.

### 1.2 `open_meteo_era5_lahore_daily_2019_2026.csv` (main weather)
- **What:** daily weather for Lahore from ERA5 reanalysis.
- **Source:** Open-Meteo Historical Weather API (`archive-api.open-meteo.com`), `models=era5`, `timezone=Asia/Karachi`, requested point 31.558 N, 74.351 E. The file header shows the grid cell used: 31.5 N, 74.25 E, elevation 220 m.
- **Variables (daily):** temperature (mean, max, min), relative humidity (mean), dew point (mean), surface pressure (mean), wind speed (mean, max), dominant wind direction, precipitation sum, rain sum, shortwave radiation sum, cloud cover (mean). Units are in the column headers.
- **Coverage:** 2019-06-01 to 2026-10-08. 2,687 rows, no missing dates. The last row (2026-10-08) is empty because of the reanalysis delay, so 2,686 usable days. The first 3 lines of the file are a metadata header (read with `skiprows=3`).
- **Quality:** agrees closely with station temperature (correlation 0.99) and dew point (0.99), reasonably with humidity (0.94), and less with wind (0.64) and rain (0.4 to 0.7 depending on the reference).
- **License / attribution:** Open-Meteo states a free non-commercial option and CC BY 4.0 for its data. Credit Open-Meteo and the underlying ERA5 data (Copernicus Climate Change Service / ECMWF) using the wording on Open-Meteo's attribution page. **TO CONFIRM** exact wording before submission.

### 1.3 `station_aqi_weather_lahore_daily_2019_2023.csv` (cross-check)
- **What:** daily Lahore PM2.5 AQI plus weather columns (temperature in °F, wind in mph, pressure in inches).
- **Source:** **TO CONFIRM.** Supplied as `lahore_aqi_2019_to_2023.csv`; the origin is not documented. Its AQI values behave like the US Diplomatic Post data (correlation 0.99 with 1.1 converted to AQI), so the AQI is probably from the same monitor (inferred). The weather source is unknown.
- **Coverage:** 2019-06-01 to 2023-11-30, 1,644 rows, no missing dates, dates formatted `dd-mm-yy`.
- **Quality:** AQI is missing on 221 days (13.4%), including a 47-day gap from 22 Oct to 7 Dec 2021. Zeros in the minimum temperature, dew point, humidity and pressure columns are invalid and should be treated as missing. Seven average pressure values are implausibly low.
- **License:** **TO CONFIRM.**

### 1.4 `modeled_cams_lahore_hourly_2023_2025.csv` (modeled comparison)
- **What:** hourly modeled pollutants and weather for Lahore.
- **Source:** **TO CONFIRM.** Supplied as `dataset.csv`. The field names (`pm2_5`, `nitrogen_dioxide`, `temperature_2m`, `pressure_msl`) match Open-Meteo's air-quality and weather APIs, which serve CAMS global atmospheric-composition data at about 0.4° (roughly 45 km) from August 2022 (probable, inferred).
- **Coverage:** 2023-01-01 00:00 to 2025-05-01 23:00, 20,448 rows, no missing hours. No rain column, and no location in the file (matches other Lahore files).
- **Quality:** modeled, not measured. Against the ground-truth daily means it reads about half as high (ratio 0.53 over 604 overlapping days) and under-reads smog peaks, although its day-to-day pattern tracks well (correlation 0.83 for Jul to Dec 2024). `aqi_category`-style columns are derived from PM2.5 and must not be used as inputs.
- **License:** **TO CONFIRM** (see Open-Meteo above if confirmed).

### 1.5 `modeled_cams_lahore_daily_2022_2026.csv`, `city_info.csv`, `data_dictionary.csv` (optional)
- **What:** daily modeled pollutants (PM2.5, PM10, CO, NO₂, SO₂, ozone, aerosol optical depth, dust, UV index, US AQI, European AQI) for Lahore. `city_info.csv` gives the city coordinates (31.558 N, 74.351 E) and `data_dictionary.csv` lists column units.
- **Source:** **TO CONFIRM.** Probably the same CAMS-based service as 1.4 (its start, August 2022, matches the service's data start). Its daily PM2.5 agrees with 1.4 (correlation 0.985).
- **Coverage:** 2022-08-01 to 2026-02-18, 1,298 rows, no missing dates. The first 3 rows are empty and one `us_aqi` value is missing.
- **Quality:** same limitations as 1.4. Against the station AQI over 394 overlapping days it correlates 0.67, averages 27 AQI points lower, and reads about 207 to 270 on days when the station reported 447 to 491. The daily `us_aqi` is an average of hourly values, not the official daily AQI.
- **Use:** only for a calibration or comparison experiment, never merged directly with the ground truth.

## 2. Derived files (reproducible)

### 2.1 `lahore_us_post_pm25_hourly_2019_2025.csv`
Built from 1.1. Columns: `timestamp_utc_period_start`, `pm25_ugm3`, `has_flag`, `timestamp_local`, `source`. Empty values stay empty (no imputation).

### 2.2 `lahore_us_post_pm25_daily_2019_2025.csv`
Built from 2.1. One row per local day with `valid_hours`, `pm25_mean_ugm3`, and `usable_day = valid_hours >= 18`. 2,038 daily rows, 1,506 usable.

These were first produced during exploration and **must be regenerated by the project's cleaning script** (`data_scripts/`) so the pipeline is reproducible. The AQI conversion from µg/m³ (which breakpoint version) is **TO DECIDE** and must be stated in the report.

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
No open, time-aligned Lahore traffic dataset was found. Traffic is represented by **calendar proxies** (weekday, weekend, public holidays, Ramadan and Eid, school terms) in a small table built by the team (**TO BUILD**). Moon-sighting can shift religious dates by a day, so note that in the report.

## 6. Planned pipeline scripts (`data_scripts/`)
1. `01_download_data.py`: downloads the API-based sources and writes `reports/data_collection_log.md` (no API keys in code; the OpenAQ key is read from the `OPENAQ_API_KEY` environment variable).
2. `02_clean_pm25.py`: cleans the OpenAQ file and builds the hourly and daily series.
3. `03_clean_weather.py`: cleans the Open-Meteo file (3-line header, empty last row).
4. `04_build_features.py`: joins PM2.5, weather and calendar proxies, then builds lag and rolling features.

## 7. Attribution and licenses (checklist)
- [ ] OpenAQ and the US Diplomatic Post data provider: confirm the license and the required credit.
- [ ] Open-Meteo and ERA5 (Copernicus / ECMWF): confirm the attribution wording.
- [ ] Origin and license of the station file and the two modeled files.
- [ ] State in the report that the modeled files are not ground truth.
