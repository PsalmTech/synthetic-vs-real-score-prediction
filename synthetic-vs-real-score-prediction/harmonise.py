"""Harmonisation code for:
"When Synthetic Benchmarks Meet Real Students: Auditing, Pooling and Externally Validating Score-Prediction Models".

Maps the four source datasets onto one schema:
    study_hours_week, attendance, parent_education, school_type, internet_access, gender, score (0-100)

Usage
-----
    from harmonise import load_all
    data = load_all("data/")          # dict with keys A, B, dev, C, D_mat, D_por, and the raw frames

Run `python harmonise.py data/` to print an audit summary of the harmonised data.
The same logic is used, cell by cell, in paper_analysis.ipynb.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

# Common three-level parental-education scale (Datasets A and B)
EDU_MAP = {
    # Dataset A
    "no formal": "secondary_or_less", "high school": "secondary_or_less",
    "diploma": "tertiary", "graduate": "tertiary",
    "post graduate": "postgraduate", "phd": "postgraduate",
    # Dataset B (lower-cased)
    "college": "tertiary", "postgraduate": "postgraduate",
}

# UCI studytime bands (weekly): 1: <2 h, 2: 2-5 h, 3: 5-10 h, 4: >10 h -> interval mid-points (open top band assumed 12 h)
UCI_STUDYTIME_HOURS = {1: 1.0, 2: 3.5, 3: 7.5, 4: 12.0}

FEATURES_DEV = ["study_hours_week", "attendance", "parent_education", "school_type", "internet_access", "gender"]
FEATURES_C = ["study_hours_week", "attendance"]                                            # shared with Dataset C
FEATURES_D = ["study_hours_week", "parent_education", "internet_access", "gender"]          # shared with UCI


def _lower_strings(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype).startswith("str"):
            d[c] = d[c].str.strip().str.lower()
    return d


def harmonise_A(raw: pd.DataFrame) -> pd.DataFrame:
    """Dataset A (Kaggle: kundanbedmutha/student-performance-dataset).
    Removes exact duplicate rows (10,000 of 25,000) and converts daily study hours to weekly hours (x 7)."""
    A = _lower_strings(raw.drop_duplicates())
    out = pd.DataFrame({
        "study_hours_week": A["study_hours"] * 7,
        "attendance": A["attendance_percentage"],
        "parent_education": A["parent_education"].map(EDU_MAP),
        "school_type": A["school_type"],
        "internet_access": A["internet_access"],
        "gender": A["gender"],
        "score": A["overall_score"].clip(0, 100),
    })
    out["source"] = "A"
    return out


def harmonise_B(raw: pd.DataFrame) -> pd.DataFrame:
    """Dataset B (Kaggle: lainguyn123/student-performance-factors).
    Uses Parental_Education_Level (NOT Parental_Involvement); clips the single score of 101 to 100."""
    B = _lower_strings(raw)
    out = pd.DataFrame({
        "study_hours_week": B["Hours_Studied"],
        "attendance": B["Attendance"],
        "parent_education": B["Parental_Education_Level"].map(EDU_MAP),   # 90 missing -> imputed inside the model pipeline
        "school_type": B["School_Type"],
        "internet_access": B["Internet_Access"],
        "gender": B["Gender"],
        "score": B["Exam_Score"].clip(0, 100),
    })
    out["source"] = "B"
    return out


def harmonise_C(raw: pd.DataFrame) -> pd.DataFrame:
    """Dataset C (Kaggle: nabeelqureshitiii/student-performance-dataset), external test only."""
    out = pd.DataFrame({
        "study_hours_week": raw["weekly_self_study_hours"],
        "attendance": raw["attendance_percentage"],
        "score": raw["total_score"].clip(0, 100),
    })
    out["source"] = "C"
    return out


def harmonise_uci(raw: pd.DataFrame, course: str) -> pd.DataFrame:
    """Dataset D (UCI Student Performance, Cortez and Silva 2008; DOI 10.24432/C5TG7T).
    G1 and G2 are never used (earlier grades of the same course would leak the outcome)."""
    out = pd.DataFrame({
        "study_hours_week": raw["studytime"].map(UCI_STUDYTIME_HOURS),
        "parent_education": np.where(np.maximum(raw["Medu"], raw["Fedu"]) >= 4, "tertiary", "secondary_or_less"),
        "internet_access": raw["internet"].str.lower(),
        "gender": raw["sex"].map({"F": "female", "M": "male"}),
        "score": raw["G3"] * 5.0,
    })
    out["course"] = course
    return out


def collapse_to_uci_levels(dev: pd.DataFrame) -> pd.DataFrame:
    """For transfer to UCI, which does not distinguish postgraduate study, merge postgraduate into tertiary."""
    d = dev.copy()
    d["parent_education"] = d["parent_education"].replace({"postgraduate": "tertiary"})
    return d


def load_all(data_dir: str = "data/") -> dict:
    raw_A = pd.read_csv(data_dir + "Student_Performance.csv")
    raw_B = pd.read_csv(data_dir + "StudentPerformanceFactors.csv")
    raw_C = pd.read_csv(data_dir + "dataset_C_nabeel.csv")
    if set(raw_C.columns) == set(raw_A.columns):
        raise ValueError("Dataset C has Dataset A's columns - wrong file (check for case-insensitive file-name clashes).")
    raw_mat = pd.read_csv(data_dir + "student-mat.csv", sep=";")
    raw_por = pd.read_csv(data_dir + "student-por.csv", sep=";")
    A, B = harmonise_A(raw_A), harmonise_B(raw_B)
    return {
        "A": A, "B": B, "dev": pd.concat([A, B], ignore_index=True),
        "C": harmonise_C(raw_C),
        "D_mat": harmonise_uci(raw_mat, "Mathematics"), "D_por": harmonise_uci(raw_por, "Portuguese"),
        "raw": {"A": raw_A, "B": raw_B, "C": raw_C, "mat": raw_mat, "por": raw_por},
    }


if __name__ == "__main__":
    d = load_all(sys.argv[1] if len(sys.argv) > 1 else "data/")
    raw = d["raw"]
    print(f"Dataset A: {len(raw['A']):,} rows, {raw['A'].duplicated().sum():,} exact duplicates -> {len(d['A']):,} unique")
    print(f"Dataset B: {len(d['B']):,} rows | pooled development set: {len(d['dev']):,}")
    print(f"Dataset C: {len(d['C']):,} rows | UCI Mathematics: {len(d['D_mat'])} | UCI Portuguese: {len(d['D_por'])}")
    for name in ["A", "B", "C", "D_mat", "D_por"]:
        x = d[name]
        r = x[["study_hours_week", "score"]].corr().iloc[0, 1]
        print(f"  {name:6s} score mean {x.score.mean():6.2f} (SD {x.score.std():5.2f}) | r(weekly study hours, score) = {r:.3f}")
