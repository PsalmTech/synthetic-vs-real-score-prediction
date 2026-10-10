# Data

The data files are **not** included in this repository. Download them from their original sources and place them here with exactly these names:

| File name in `data/` | Dataset | Source |
|---|---|---|
| `Student_Performance.csv` | A | https://www.kaggle.com/datasets/kundanbedmutha/student-performance-dataset |
| `StudentPerformanceFactors.csv` | B | https://www.kaggle.com/datasets/lainguyn123/student-performance-factors |
| `dataset_C_nabeel.csv` | C | https://www.kaggle.com/datasets/nabeelqureshitiii/student-performance-dataset (rename the downloaded CSV) |
| `student-mat.csv`, `student-por.csv` | D (UCI) | https://archive.ics.uci.edu/dataset/320/student+performance (DOI 10.24432/C5TG7T; unzip the inner `student.zip`) |

**Important:** Windows file names are not case-sensitive, so `student_performance.csv` and `Student_Performance.csv` are the same file there. Dataset C must therefore be saved under a different name (`dataset_C_nabeel.csv`). `harmonise.py` raises an error if Dataset C turns out to have Dataset A's columns.

All files accessed 9 October 2026. The UCI data are licensed CC BY 4.0 (Cortez and Silva, 2008).
