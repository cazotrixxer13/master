# DILA Entity Matching Project (WS 2025/26)

This project implements a complete **Entity Matching (EM)** pipeline for bibliographic data from DBLP and Google Scholar using Python.  
It includes data cleaning, deduplication, blocking, TF-IDF similarity, feature engineering, and supervised ML models evaluated against a gold standard mapping.

All experiments were executed in a **conda environment with Python 3.11.13**.

---

## Environment Setup

### 1. Create and activate conda environment

```bash
conda create -n "dila" python=3.11.13
conda activate dila
```

### 2. Install the required Python packages

The file **requirements.txt** contains the exact pinned versions used for this project.

```bash
pip install -r requirements.txt
```

### Running the Project

The entire pipeline is implemented in the notebook **final1.ipynb**. To run it:

```bash
jupyter notebook final1.ipynb
```

Or alternatively, in VS Code, open the file and select the dila environment which was previously created as python kernel when prompted.

---

## Pipeline Overview

The notebook executes the pipeline in the following order:

1. **Data loading & normalization**
2. **Deduplication per source**
3. **Blocking & candidate generation**
4. **TF-IDF similarity computation**
5. **Threshold evaluation (Task 01)**
6. **Feature extraction**
7. **Supervised ML models (RF + SVM)**
8. **Model evaluation (Task 02)**

All results are fully **reproducible** when executed end-to-end with the provided environment.