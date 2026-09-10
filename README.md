# Machine-Learning-Classification

Leakage-aware binary classification workflow for high-dimensional numeric data such as transcriptomics, proteomics, and biomarker panels. The repository provides parallel R and Python implementations and separates feature discovery, model-family comparison, and final model freezing.

## Three questions

### Q1. What is the optimized feature combination for prediction?

Use nested cross-validation with feature selection **inside the inner training folds**. The default selector is a univariate binary F / point-biserial-correlation ranking, and the number of selected features is treated as a hyperparameter rather than fixed at 30. Elastic Net is used as the reference model because it is regularized and appropriate for correlated, high-dimensional predictors.

Outputs include unbiased outer-fold performance, selected-feature stability, the distribution of optimal feature counts, and a full-development-data candidate feature panel.

Scripts:

- `python/q1_optimize_features.py`
- `R/q1_optimize_features.R`

### Q2. How do we train an optimized model that can be frozen and transferred?

After the model family has been chosen, tune the final pipeline on the full development cohort, lock the selected features, fit preprocessing on the development data only, refit the chosen classifier, and serialize the complete artifact. The transfer script requires the locked features and applies the stored imputation/scaling/model without refitting on the new cohort.

Scripts:

- `python/q2_train_frozen_model.py` → `frozen_model.joblib`
- `python/apply_frozen_model.py`
- `R/q2_train_frozen_model.R` → `frozen_model.rds`
- `R/apply_frozen_model.R`

**Do not report the full-data refit score as model performance.** Performance comes from Q3 nested CV and, ideally, an untouched external validation cohort.

### Q3. Which machine-learning model is most suitable?

Compare model families on identical outer folds. For every outer fold, feature selection, feature-count tuning, preprocessing, and model hyperparameter tuning are restricted to the training partition. The untouched outer fold is used only once for evaluation.

Core models:

- Elastic Net logistic regression
- Random Forest
- RBF Support Vector Machine
- Gaussian Naive Bayes

The primary selection metric is ROC AUC. PR AUC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, and Brier score are also reported. A one-standard-error rule is used to avoid choosing a more complex model for a negligible AUC difference.

Scripts:

- `python/q3_compare_models.py`
- `R/q3_compare_models.R`

## Recommended execution order

The scientific questions are labeled Q1/Q2/Q3 above, but the executable order should be:

**Q1 feature optimization → Q3 model comparison → Q2 frozen-model training → external transfer/validation**

Q2 must come after Q3 in execution because the final model family should be chosen before the frozen artifact is trained.

## Why the original workflow was changed

The original root-level scripts are retained as legacy examples, but they should not be used for final model selection. The main issues were:

1. A fixed 30-feature panel was selected before inner model tuning, so inner validation did not fully include the feature-selection operation.
2. R used a t-test selector while Python used mutual information, so results were not methodologically comparable.
3. Only accuracy was emphasized. Accuracy alone can be misleading under class imbalance and does not evaluate probability ranking or calibration.
4. Preprocessing was inconsistent across model families and languages.
5. The final frozen model was not separated from unbiased performance estimation.
6. A 1D CNN was applied to arbitrarily ordered genes. Convolution assumes meaningful local adjacency along the feature axis, which ordinary tabular gene-expression columns do not provide.
7. Package installation was mixed into analysis code, reducing reproducibility and making production execution brittle.

## Input format

Use a CSV with one row per sample, one binary outcome column, optional sample ID, and numeric feature columns.

```text
sample_id,GeneA,GeneB,GeneC,label
S001,0.32,1.15,-0.42,Case
S002,-0.10,0.87,0.91,Control
```

## Python

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Q1:

```bash
python python/q1_optimize_features.py \
  --data data.csv --label label --id-column sample_id --positive Case
```

Q3:

```bash
python python/q3_compare_models.py \
  --data data.csv --label label --id-column sample_id --positive Case
```

Q2, using the Q3 recommendation:

```bash
python python/q2_train_frozen_model.py \
  --data data.csv --label label --id-column sample_id --positive Case \
  --recommended-model-file results/q3_model_comparison/recommended_model.txt
```

Transfer:

```bash
python python/apply_frozen_model.py \
  --model artifacts/python_frozen_model/frozen_model.joblib \
  --data external_cohort.csv --id-column sample_id \
  --output external_predictions.csv
```

## R

Install dependencies once:

```bash
Rscript R/install_packages.R
```

Q1:

```bash
Rscript R/q1_optimize_features.R \
  --data data.csv --label label --id-column sample_id --positive Case
```

Q3:

```bash
Rscript R/q3_compare_models.R \
  --data data.csv --label label --id-column sample_id --positive Case
```

Q2:

```bash
Rscript R/q2_train_frozen_model.R \
  --data data.csv --label label --id-column sample_id --positive Case \
  --recommended-model-file results/q3_model_comparison/recommended_model.txt
```

Transfer:

```bash
Rscript R/apply_frozen_model.R \
  --model artifacts/R_frozen_model/frozen_model.rds \
  --data external_cohort.csv --id-column sample_id \
  --output external_predictions.csv
```

## Feature-count search

Default candidate feature counts are `5, 10, 20, 30, 50, 100, all`, truncated to the available number of predictors. Override with, for example:

```bash
--feature-counts 10,20,30,40,57,100
```

This is preferable to hard-coding one panel size before cross-validation.

## Cross-validation defaults

The default is 5 outer folds and 4 inner folds. For a final analysis, consider repeated nested CV if compute permits, and always prioritize a genuinely independent cohort for transfer validation. Use identical sample-level grouping across folds when repeated measures, longitudinal samples, families, sites, or other non-independent units are present; ordinary stratified folds are not sufficient in those settings.

## Deep learning

CNN is not part of the core benchmark. For ordinary omics matrices, the feature axis usually lacks a local spatial order, and small sample sizes strongly favor regularized classical models. If a neural-network benchmark is scientifically justified, add it as a separate module and keep its preprocessing and tuning inside the same nested-CV framework. For current R Keras workflows, use `keras3` rather than the older `keras` package.

## Repository layout

```text
R/
  common.R
  install_packages.R
  q1_optimize_features.R
  q2_train_frozen_model.R
  q3_compare_models.R
  apply_frozen_model.R
python/
  common.py
  q1_optimize_features.py
  q2_train_frozen_model.py
  q3_compare_models.py
  apply_frozen_model.py
requirements.txt
```

The root-level `machine_learning_classification.R`, `machine_learning_classification.py`, and `Nested cross validation structure` are retained only for historical reference.
