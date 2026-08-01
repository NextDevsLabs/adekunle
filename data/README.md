# Data

Place the NHS England Monthly A&E Time Series workbook here:

```text
Monthly-AE-Time-Series-March-2026-F5ldj2 (1).xls
```

## Download

1. Open: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/
2. Download the **Monthly A&E Time Series** release used in the dissertation (March 2026).
3. Save the `.xls` file into this `data/` folder with the filename above  
   (or update `DEFAULT_DATA_PATH` in `src/data_loader.py`).

The file is published under the Open Government Licence.

## Sheets used

- **Activity** — attendances and emergency admissions  
- **Performance** — four-hour performance and waits over four hours  
- **Booking** — booked attendances (from August 2020)

## Fallback

If the Excel file is absent, `src/data_loader.py` will load  
`outputs/tables/master_dataset.csv` (the pre-merged dataset from this study).
