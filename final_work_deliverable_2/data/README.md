# Data dependency

The modelling table is intentionally **not** copied into this package:

```text
data/processed/features_daily.csv
```

It is generated data, is not committed to Git, and should be restored from the
shared Drive or regenerated from the Phase 2 raw files.

Shared Drive folder:

```text
https://drive.google.com/drive/folders/1gRY4v3sCW90rdJ97O1RFDDux_Dxpd89Y
```

## Expected checks after restoring the file

From the repository root:

```powershell
python -m src.check_setup
```

Expected key values:

- 2,113 daily rows; 2,113-day span; 0 missing dates.
- 0 bad next-day PM2.5 values; 0 bad lag-1 PM2.5 values.
- 0 bad next-day category values.
- 1,022 train rows, 250 validation rows, 223 test rows.
- 917/244/222 scorable rows.
- Six classification classes; Good merged into Moderate.
- Train 2019-05-23 → 2023-06-30.
- Validation 2023-07-14 → 2024-06-03.
- Test 2024-07-08 → 2025-02-16.

The report's results were generated from the final output copies in
`../metrics/`; those copies are frozen evidence and do not change when the data
is re-downloaded.
