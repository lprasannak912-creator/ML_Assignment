# Project Architecture

```text
data/wdbc.data
       |
       v
src/data.py  -- validates the raw WDBC file and separates features/label
       |
       v
train.py     -- your training, test split, metrics, and model-saving work
       |                         |
       v                         v
models/*.joblib             test_data.csv
                                 |
                                 v
app.py       -- loads a saved model, accepts labeled test CSV, reports metrics
```

## Responsibilities

- `src/data.py`: column names, data loading, basic validation, and feature/target separation.
- `train.py`: to be completed with a stratified split, five classifiers, required metrics, and saved model pipelines.
- `app.py`: interface shell. It does not train models; the evaluation workflow is a TODO until saved models exist.
- `data/`: raw source data and the labeled held-out test CSV. Do not upload the full training dataset to the free app.
- `models/`: saved pipelines created by your training run.

## Important boundary

Run the assignment workflow in BITS Virtual Lab and capture your own lab screenshot. Use your own GitHub and Streamlit accounts for publishing. This scaffold is for learning and is not a completed assignment.