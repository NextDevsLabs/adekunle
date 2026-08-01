# NHS A&E Four-Hour Performance Forecasting

Replication code for the MSc Data Science dissertation:

**Predicting Emergency Department Waiting Times Using Time-Series and Machine Learning Models**  
UEL-DS-7010 | University of East London / UNICAF

This repository compares **ARIMA**, **Prophet**, **Random Forest**, and **LSTM** on monthly NHS England A&E four-hour performance (November 2010 – March 2026).

---

## Repository structure

```text
.
├── data/                          # Raw NHS Excel workbook (+ data notes)
├── notebooks/
│   ├── 01_eda_and_data_preparation.ipynb
│   └── 02_modeling_and_evaluation.ipynb
├── outputs/
│   ├── figures/                   # Charts used in the dissertation
│   └── tables/                    # Metrics, DM tests, feature importance
├── src/
│   ├── data_loader.py             # Load & merge Activity / Performance / Booking
│   ├── features.py                # Flags, lags, splits, scaling
│   ├── metrics.py                 # MAE, RMSE, MAPE, R², Diebold–Mariano
│   └── models.py                  # ARIMA, Prophet, RF, LSTM
├── run_pipeline.py                # End-to-end script
├── requirements.txt
└── README.md
```

---

## Setup

1. Clone this repository.
2. Create a virtual environment (recommended):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Confirm the data file is present (see `data/README.md`):

```text
data/Monthly-AE-Time-Series-March-2026-F5ldj2 (1).xls
```

If the Excel file is missing, the pipeline can fall back to `outputs/tables/master_dataset.csv` (pre-merged).

---

## Reproduce the full analysis

From the **repository root**:

```bash
python run_pipeline.py
```

This will:

1. Load and merge the NHS sheets  
2. Build features and chronological train / validation / test splits  
3. Train ARIMA, Prophet, Random Forest, and LSTM  
4. Score the test set (Apr 2025 – Mar 2026)  
5. Run Diebold–Mariano pairwise tests  
6. Run CRS-exclusion sensitivity analysis  
7. Write figures to `outputs/figures/` and tables to `outputs/tables/`

### Notebooks (optional)

```bash
jupyter notebook notebooks/
```

- **01** — EDA and data preparation  
- **02** — Runs `run_pipeline.main()` and displays results  

---

## Train / validation / test splits

| Split      | Period                      | Months |
|------------|-----------------------------|--------|
| Train      | Nov 2010 – Mar 2023         | 149    |
| Validation | Apr 2023 – Mar 2025         | 24     |
| Test       | Apr 2025 – Mar 2026         | 12     |

Target variable: `pct_within_4h_all` (share of attendances seen within four hours).

---

## Expected headline results

After a successful run, `outputs/tables/model_comparison.csv` should rank models similarly to:

| Model         | Approx. test MAE (pp) |
|---------------|------------------------|
| Prophet       | 0.78                   |
| Random Forest | 0.90                   |
| LSTM          | 1.03                   |
| ARIMA         | 2.12                   |

LSTM scores can vary slightly across TensorFlow / OS versions even with `random_state` / seed `42`. ARIMA, Prophet, and Random Forest should match closely.

---

## Licence / data

- **Code:** provided for academic replication of this dissertation.  
- **Data:** NHS England Monthly A&E Time Series, published under the [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).  
  Source: https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/

---

## Citation

If you use this code, please cite the dissertation and NHS England as the data source.
