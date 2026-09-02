# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts `posted_rate` for freight loads using XGBoost trained on a distance-normalized,
log-transformed target (`log1p(rate_per_mile)`, rescaled by `distance` at prediction time).

## Repo structure

```
.
├── README.md
├── report.docx
├── requirements.txt
├── solution.ipynb        # full pipeline: EDA -> features -> split -> tuning -> predictions
└── data/                       # not included in this repo — see Setup below
    ├── train-test.csv
    ├── validation.csv
    ├── validation-predictions-template.csv
    └── december-chart-inputs.csv
```

## Setup

1. Place the four provided data files in `data/` (paths above). They're not committed to this
   repo — copy them from wherever the assessment provided them.
2. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. **Reproducibility note:** XGBoost's exact output can vary slightly by library version even
   with `random_state` fixed. If you need bit-for-bit reproduction of the numbers in the report,
   run `pip freeze | grep -iE "xgboost|scikit-learn"` in this environment and pin those exact
   versions in `requirements.txt` before submitting.

## Run

1. Open and run all cells in `solution.ipynb` top to bottom (`jupyter notebook` or
   `jupyter nbconvert --to notebook --execute`). This will:
   - Clean and explore `data/train-test.csv`
   - Engineer features and split train/test chronologically (85/15 by date)
   - Impute missing values using train-only statistics
   - Report held-out MAE/RMSE and segment-level error
   - Write `validation_predictions.csv` (from `data/validation.csv`)
   - Write a completed `december-chart-inputs.csv` (from `data/december-chart-inputs.csv`)

2. From the repo root, run the provided scorer against those two output files:
   ```bash
   python3 score.py \
     --predictions validation_predictions.csv \
     --december-predictions december-chart-inputs.csv
   ```
   This validates both files and writes `scorer_results/candidate_december.png`.

## Approach summary

- **Split:** time-based (train on earlier dates, test on later ones) to mirror real deployment
  and avoid leaking time-correlated market conditions across the split.
- **Target:** `log1p(rate_per_mile)` rather than raw `posted_rate` — normalizing by distance
  before modeling fixed two segment-level weak spots (reefer equipment, short-haul loads).
- **Imputation:** mean-filled using train-only statistics, reused unchanged on test/validation/
  December — never refit on data the model wouldn't have at prediction time.
- **December chart:** `market_index`/`quote_signal` for the 31 fixed-lane rows are filled using
  the real November/December average observed in `validation.csv`, not the whole-year training
  mean (training data has no December rows, and the whole-year mean is biased by the spring/
  summer seasonal peak).

Full detail, including the hyperparameter search and error analysis, is in
`solution.ipynb` and in the submitted report.
