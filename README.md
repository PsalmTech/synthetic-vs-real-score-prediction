# When Synthetic Benchmarks Meet Real Students

Code and analysis notebook for the paper:

> Ayogu, B. A., Ojeleke, S. A., Adeniyi, D. A., Ajala, O. H., and Adewoyin, O. *When Synthetic Benchmarks Meet Real Students: Auditing, Pooling and Externally Validating Score-Prediction Models.* Manuscript submitted to the Journal of Educational Data Mining.

The study audits and pools two public student-performance datasets, compares Random Forest, XGBoost and a stacking ensemble against mean and linear-regression baselines, and tests transfer to an independent synthetic dataset (1,000,000 records) and to real students (UCI Student Performance data).

## Contents

| File | What it is |
|---|---|
| `paper_analysis.ipynb` | The complete analysis, already executed, with every table and figure in the paper |
| `harmonise.py` | Stand-alone harmonisation code that maps all four datasets to one schema (identical to the notebook's harmonisation) |
| `data/README.md` | Where to download each dataset and what to name it |
| `requirements.txt` | Package versions used |

## How to reproduce

```bash
git clone <this repository URL>
cd synthetic-vs-real-score-prediction
pip install -r requirements.txt
# download the five CSV files into data/ as described in data/README.md
python harmonise.py data/          # quick audit of the harmonised data
jupyter notebook paper_analysis.ipynb
```

Running the whole notebook takes about 25 minutes on a two-core laptop (the stacking cross-validation and learning curves are the slow parts). All random processes use seed 42. Figures are written to `figures/`.

## Harmonisation in brief

* **Dataset A:** 10,000 exact duplicate rows removed (25,000 → 15,000); study hours per day × 7 → per week.
* **Dataset B:** `Parental_Education_Level` mapped to the common scale (`Parental_Involvement` is a different construct and is not used).
* **Parental education:** common three levels (secondary or less / tertiary / postgraduate); two levels for UCI.
* **UCI:** `studytime` bands → weekly-hour mid-points (1, 3.5, 7.5, 12); `G3` × 5 → 0–100; `G1` and `G2` never used.
* The dataset label is kept for reporting only and is **never** used as a predictor.

## Licence

Code: MIT (see `LICENSE`). Data are not redistributed; see `data/README.md` for their sources and licences.
