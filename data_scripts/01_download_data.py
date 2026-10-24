#!/usr/bin/env python3
"""
01_download_data.py -- SmogShield Lahore, Deliverable 1, Phase 2 (Collection)

Re-downloads the public sources used by the project and appends a provenance
section to reports/data_collection_log.md.

Uses only the Python standard library (no pip installs required).

Sources
-------
1. OpenAQ v3, sensor 25135 (US Diplomatic Post, Lahore)   -> raw/ground_truth/
2. Open-Meteo Historical Weather API (ERA5, daily)         -> raw/weather/
3. Open-Meteo Air Quality API (CAMS, hourly)               -> raw/validation/
4. Zenodo DOI 10.5281/zenodo.18673725 (Kulkarni et al.)    -> raw/validation/
5. GitHub taimoorbarlas15/Lahore_Air_Quality (station)     -> raw/validation/

Usage
-----
Windows (cmd):
    set OPENAQ_API_KEY=your_key_here
    python data_scripts\\01_download_data.py

bash / PowerShell:
    export OPENAQ_API_KEY=your_key_here      # bash
    $env:OPENAQ_API_KEY="your_key_here"      # PowerShell
    python data_scripts/01_download_data.py

The OpenAQ key is read from the environment only. It is never written to disk
and must never be committed to the repository.

Only re-download what you need, for example:
    python data_scripts/01_download_data.py --only zenodo --only station
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# In the repository the script lives in <repo>/data_scripts and the data folder
# is <repo>/data. In the course working folder the script lives inside the data
# folder itself (<data>/data_scripts), so fall back to ROOT when ROOT/data is absent.
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT
RAW = DATA / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

LOG: list[dict] = []


# --------------------------------------------------------------------------- #
# HTTP helpers (standard library only)
# --------------------------------------------------------------------------- #
def http_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> bytes:
    """GET a URL and return the body, retrying transient network failures.

    Large Open-Meteo responses occasionally end early (IncompleteRead); a retry
    or a smaller date chunk usually fixes it.
    """
    if params:
        sep = "&" if "?" in url else "?"
        url = url + sep + urllib.parse.urlencode(params)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            request.add_header("User-Agent", "SmogShield-data-collection/1.0")
            request.add_header("Accept-Encoding", "identity")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, http.client.IncompleteRead, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise last_error  # type: ignore[misc]


def http_get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 120) -> dict:
    return json.loads(http_get(url, params=params, headers=headers, timeout=timeout).decode("utf-8"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_data_rows(path: Path, skip_header_lines: int = 1) -> int:
    """Number of data rows in a CSV (total lines minus header/metadata lines)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        total = sum(1 for _ in fh)
    return max(total - skip_header_lines, 0)


def record(name: str, url: str, path: Path, note: str = "", skip_header_lines: int = 1) -> None:
    LOG.append(
        {
            "name": name,
            "url": url,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": count_data_rows(path, skip_header_lines),
            "note": note,
        }
    )
    print(f"  saved {path} ({path.stat().st_size:,} bytes)")


# --------------------------------------------------------------------------- #
# 1. OpenAQ v3 -- sensor 25135
# --------------------------------------------------------------------------- #
OPENAQ_BASE = "https://api.openaq.org/v3"
OPENAQ_SENSOR = 25135
OPENAQ_START = date(2019, 5, 22)
OPENAQ_END = date(2025, 3, 4)

# Column order chosen to reproduce the delivered raw file exactly.
OPENAQ_COLUMNS = [
    "value",
    "coordinates",
    "flagInfo.hasFlags",
    "parameter.id",
    "parameter.name",
    "units",
    "parameter.displayName",
    "period.label",
    "period.interval",
    "date_utc",
    "date_local",
    "period.datetimeTo.utc",
    "period.datetimeTo.local",
    "summary.min",
    "summary.q02",
    "summary.q25",
    "summary.median",
    "summary.q75",
    "summary.q98",
    "summary.max",
    "summary.avg",
    "summary.sd",
    "coverage.expectedCount",
    "coverage.expectedInterval",
    "coverage.observedCount",
    "coverage.observedInterval",
    "coverage.percentComplete",
    "coverage.percentCoverage",
    "coverage.datetimeFrom.utc",
    "coverage.datetimeFrom.local",
    "coverage.datetimeTo.utc",
    "coverage.datetimeTo.local",
    "sensor_id",
]


def _flatten_measurement(m: dict) -> dict:
    period = m.get("period") or {}
    dt_from = period.get("datetimeFrom") or {}
    dt_to = period.get("datetimeTo") or {}
    cov = m.get("coverage") or {}
    summ = m.get("summary") or {}
    par = m.get("parameter") or {}
    c_from = cov.get("datetimeFrom") or {}
    c_to = cov.get("datetimeTo") or {}
    flag = m.get("flagInfo") or {}

    return {
        "value": m.get("value"),
        "coordinates": m.get("coordinates"),
        "flagInfo.hasFlags": flag.get("hasFlags"),
        "parameter.id": par.get("id"),
        "parameter.name": par.get("name"),
        "units": par.get("units"),
        "parameter.displayName": par.get("displayName"),
        "period.label": period.get("label"),
        "period.interval": period.get("interval"),
        "date_utc": dt_from.get("utc"),
        "date_local": dt_from.get("local"),
        "period.datetimeTo.utc": dt_to.get("utc"),
        "period.datetimeTo.local": dt_to.get("local"),
        "summary.min": summ.get("min"),
        "summary.q02": summ.get("q02"),
        "summary.q25": summ.get("q25"),
        "summary.median": summ.get("median"),
        "summary.q75": summ.get("q75"),
        "summary.q98": summ.get("q98"),
        "summary.max": summ.get("max"),
        "summary.avg": summ.get("avg"),
        "summary.sd": summ.get("sd"),
        "coverage.expectedCount": cov.get("expectedCount"),
        "coverage.expectedInterval": cov.get("expectedInterval"),
        "coverage.observedCount": cov.get("observedCount"),
        "coverage.observedInterval": cov.get("observedInterval"),
        "coverage.percentComplete": cov.get("percentComplete"),
        "coverage.percentCoverage": cov.get("percentCoverage"),
        "coverage.datetimeFrom.utc": c_from.get("utc"),
        "coverage.datetimeFrom.local": c_from.get("local"),
        "coverage.datetimeTo.utc": c_to.get("utc"),
        "coverage.datetimeTo.local": c_to.get("local"),
        "sensor_id": m.get("sensorsId"),
    }


def download_openaq() -> Path | None:
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        print("  [skip] OPENAQ_API_KEY is not set; cannot re-download OpenAQ.")
        return None

    out = RAW / "ground_truth" / "openaq_us_post_25135_hourly.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = {"X-API-Key": key}
    rows: list[dict] = []
    endpoint = f"{OPENAQ_BASE}/sensors/{OPENAQ_SENSOR}/measurements/hourly"

    chunk_start = OPENAQ_START
    while chunk_start <= OPENAQ_END:
        chunk_end = min(chunk_start + timedelta(days=89), OPENAQ_END)
        page = 1
        while True:
            params = {
                "datetime_from": f"{chunk_start.isoformat()}T00:00:00Z",
                "datetime_to": f"{chunk_end.isoformat()}T23:59:59Z",
                "limit": 1000,
                "page": page,
            }
            try:
                payload = http_get_json(endpoint, params=params, headers=headers, timeout=120)
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    raise SystemExit("OpenAQ rejected the API key (HTTP 401). Check OPENAQ_API_KEY.")
                raise
            results = payload.get("results") or []
            if not results:
                break
            rows.extend(_flatten_measurement(m) for m in results)
            total_pages = (payload.get("meta") or {}).get("totalPages") or 1
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.25)
        print(f"  chunk {chunk_start} -> {chunk_end}: {len(rows):,} rows so far")
        chunk_start = chunk_end + timedelta(days=1)

    if not rows:
        print("  [warn] OpenAQ returned no rows.")
        return None

    rows.sort(key=lambda r: r.get("date_utc") or "")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OPENAQ_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    record("OpenAQ v3 sensor 25135 hourly", endpoint, out)
    return out


# --------------------------------------------------------------------------- #
# 2. Open-Meteo ERA5 daily weather
# --------------------------------------------------------------------------- #
WEATHER_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "precipitation_sum",
    "rain_sum",
    "shortwave_radiation_sum",
    "cloud_cover_mean",
]


def download_weather() -> Path:
    out = RAW / "weather" / "open_meteo_era5_lahore_daily_2019_2026.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 31.558,
        "longitude": 74.35071,
        "start_date": "2019-06-01",
        "end_date": "2026-10-07",
        "daily": ",".join(WEATHER_VARS),
        "models": "era5",
        "timezone": "Asia/Karachi",
    }
    payload = http_get_json(url, params=params, timeout=180)
    daily = payload["daily"]
    units = payload["daily_units"]

    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n")
        fh.write(
            f'{payload["latitude"]},{payload["longitude"]},{payload["elevation"]},'
            f'{payload["utc_offset_seconds"]},{payload["timezone"]},{payload["timezone_abbreviation"]}\n'
        )
        fh.write("\n")
        fh.write("time," + ",".join(f"{v} ({units[v]})" for v in WEATHER_VARS) + "\n")
        for i, stamp in enumerate(daily["time"]):
            vals = ["" if daily[v][i] is None else daily[v][i] for v in WEATHER_VARS]
            fh.write(stamp + "," + ",".join(str(v) for v in vals) + "\n")

    record(
        "Open-Meteo ERA5 daily weather",
        url,
        out,
        note="3 metadata lines + 1 header line (read with skiprows=3)",
        skip_header_lines=4,
    )
    return out


# --------------------------------------------------------------------------- #
# 3. Open-Meteo Air Quality (CAMS) hourly
# --------------------------------------------------------------------------- #
CAMS_POLLUTANTS = [
    "pm2_5",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
]
CAMS_WEATHER = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "pressure_msl",
]


def download_cams_hourly() -> Path:
    out = RAW / "validation" / "modeled_cams_lahore_hourly_2023_2025.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    wx_url = "https://archive-api.open-meteo.com/v1/archive"
    start = date(2023, 1, 1)
    end = date(2025, 5, 1)

    # stamp -> list of values, so chunked/overlapping requests merge cleanly
    rows: dict[str, list[object]] = {}
    chunk = start
    while chunk <= end:
        chunk_end = min(chunk + timedelta(days=179), end)
        common = {
            "latitude": 31.558,
            "longitude": 74.35071,
            "start_date": chunk.isoformat(),
            "end_date": chunk_end.isoformat(),
            "timezone": "Asia/Karachi",
        }

        aq_payload = http_get_json(
            aq_url,
            params={**common, "hourly": ",".join(CAMS_POLLUTANTS), "models": "cams_global"},
            timeout=180,
        )
        wx_payload = http_get_json(
            wx_url,
            params={**common, "hourly": ",".join(CAMS_WEATHER), "models": "era5"},
            timeout=180,
        )

        aq_hourly = aq_payload.get("hourly") or {}
        wx_hourly = wx_payload.get("hourly") or {}
        wx_index = {stamp: i for i, stamp in enumerate(wx_hourly.get("time", []))}

        def value_at(hourly: dict, key: str, index: int | None) -> object:
            column = hourly.get(key)
            if index is None or not column or index >= len(column):
                return None
            return column[index]

        for i, stamp in enumerate(aq_hourly.get("time", [])):
            values: list[object] = [value_at(aq_hourly, c, i) for c in CAMS_POLLUTANTS]
            j = wx_index.get(stamp)
            values += [value_at(wx_hourly, c, j) for c in CAMS_WEATHER]
            rows[stamp] = values

        print(f"  chunk {chunk} -> {chunk_end}: {len(rows):,} hourly rows so far")
        chunk = chunk_end + timedelta(days=1)

    columns = CAMS_POLLUTANTS + CAMS_WEATHER
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("time," + ",".join(columns) + "\n")
        for stamp in sorted(rows):
            fh.write(stamp + "," + ",".join("" if v is None else str(v) for v in rows[stamp]) + "\n")

    record(
        "Open-Meteo CAMS hourly (air quality + weather)",
        f"{aq_url} + {wx_url}",
        out,
        note="Equivalent reproducible source for the delivered Kaggle file (aadbutt/dataset)",
    )
    return out


# --------------------------------------------------------------------------- #
# 4. Zenodo / Kaggle dataset lahore-pk-1172451
# --------------------------------------------------------------------------- #
ZENODO_URL = "https://zenodo.org/records/18673725/files/lahore-pk-1172451.zip?download=1"
ZENODO_FILES = {
    "air_quality_historical.csv": "modeled_cams_lahore_daily_2022_2026.csv",
    "city_info.csv": "city_info.csv",
    "data_dictionary.csv": "data_dictionary.csv",
}


def download_zenodo() -> Path:
    out_dir = RAW / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    content = http_get(ZENODO_URL, timeout=180)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for src, dst in ZENODO_FILES.items():
            target = out_dir / dst
            target.write_bytes(archive.read(src))
            record(f"Zenodo {src}", ZENODO_URL, target, note="CC0-1.0")
    return out_dir


# --------------------------------------------------------------------------- #
# 5. GitHub station file (AQI + weather, daily)
# --------------------------------------------------------------------------- #
GITHUB_STATION_URL = (
    "https://raw.githubusercontent.com/taimoorbarlas15/Lahore_Air_Quality/main/"
    "lahore_aqi_2019_to_2023.csv"
)


def download_station() -> Path:
    out = RAW / "validation" / "station_aqi_weather_lahore_daily_2019_2023.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(http_get(GITHUB_STATION_URL, timeout=120))
    record(
        "GitHub station AQI + weather",
        GITHUB_STATION_URL,
        out,
        note="No license stated in the repository",
    )
    return out


# --------------------------------------------------------------------------- #
# Log + CLI
# --------------------------------------------------------------------------- #
def write_log() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "",
        "## Download run",
        "",
        f"- Run at: **{stamp}**",
        f"- Host: `{platform.node()}` / Python {platform.python_version()}",
        f"- Data root: `{DATA}`",
        "",
        "| Dataset | Path | Rows | Bytes | SHA-256 | Source |",
        "|---|---|---:|---:|---|---|",
    ]
    for entry in LOG:
        lines.append(
            f'| {entry["name"]} | `{entry["path"]}` | {entry["rows"]:,} | '
            f'{entry["bytes"]:,} | `{entry["sha256"]}` | {entry["url"]} |'
        )
    lines.append("")
    with open(REPORTS / "data_collection_log.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


SOURCES = {
    "openaq": download_openaq,
    "weather": download_weather,
    "cams-hourly": download_cams_hourly,
    "zenodo": download_zenodo,
    "station": download_station,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="SmogShield Lahore data collection script (standard library only).")
    parser.add_argument(
        "--only",
        choices=list(SOURCES),
        action="append",
        help="Download only the named source(s). Repeatable. Default: all.",
    )
    args = parser.parse_args()
    wanted = args.only or list(SOURCES)

    print(f"Data root : {DATA}")
    print(f"Log file  : {REPORTS / 'data_collection_log.md'}")

    for name in wanted:
        print(f"\n== {name} ==")
        try:
            SOURCES[name]()
        except Exception as exc:  # noqa: BLE001 - keep going with the other sources
            print(f"  [error] {name}: {exc}")

    if LOG:
        write_log()
        print(f"\nLogged {len(LOG)} file(s).")
    else:
        print("\nNothing downloaded; log unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
