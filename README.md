# WDBC Classification Project

## Problem statement

This classifier predicts whether a breast tumor sample is benign or malignant based on 30 numeric features extracted from a digitized breast mass image. The task is binary classification because there are only two possible outcomes in the dataset: benign (`B`) and malignant (`M`).

## Dataset description

The Wisconsin Diagnostic Breast Cancer (WDBC) dataset contains 569 samples of breast cancer cases. Each record includes 30 real-valued features describing characteristics of the cell nucleus, such as radius, texture, perimeter, area, and smoothness. The target variable is the diagnosis label, where `B` indicates benign and `M` indicates malignant. The dataset is sourced from the UCI Machine Learning Repository, specifically the Wisconsin Diagnostic Breast Cancer dataset.

## GitHub repository link

Repository link: not published in this local workspace copy.

## Models and evaluation

The models were evaluated on a stratified 80/20 split with malignant treated as the positive class.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.8947 | 0.8750 | 0.9412 | 0.7619 | 0.8421 | 0.7741 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

- Logistic Regression: Strong and balanced performance with excellent AUC and a well-rounded precision/recall trade-off.
- Decision Tree: Lower performance than other classifiers, especially in recall and AUC.
- KNN: Good overall performance with strong precision, but slightly behind logistic regression and SVM.
- Gaussian Naive Bayes: Competitive AUC but weaker precision/recall balance than the top models.
- Support Vector Machine: Best overall performance with the highest accuracy and the strongest F1 and MCC values.
- Overall winner and reason: Support Vector Machine is the overall winner because it gave the best accuracy, F1, and MCC while maintaining very high AUC.

## Run locally

1. Install the packages listed in `requirements.txt`.
2. Put the raw file at `data/wdbc.data`.
3. Run `python train.py` to create model files and a labeled hold-out `data/test_data.csv`.
4. Run `streamlit run app.py` and upload a labeled CSV to evaluate the saved model.

## Links for submission

- GitHub repository: not published in this environment
- Live Streamlit app: http://localhost:8501/

## Limitations

This is an educational classification exercise, not a medical diagnostic system. Predictions must not be used for clinical decisions.