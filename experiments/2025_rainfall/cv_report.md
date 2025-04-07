# CV Reports

## Model Performance Summary

| Model | Mean AUC | Std AUC | Mean Precision | Mean Recall | Mean F1 |
|-------|---------|---------|----------------|------------|--------|
| LR_Original | 0.9003 | 0.0107 | 0.8924 | 0.9410 | 0.9160 |
| XGB_Original | 0.9095 | 0.0119 | 0.8802 | 0.9624 | 0.9193 |
| LR_RFECV | 0.8999 | 0.0107 | 0.8914 | 0.9425 | 0.9161 |
| XGB_RFECV | 0.9482 | 0.0076 | 0.9154 | 0.9518 | 0.9332 |

## LR_Original - All Folds Confusion Matrices

![LR_Original Fold 1](./cv_plots/LR_Original_fold1.png) ![LR_Original Fold 2](./cv_plots/LR_Original_fold2.png)

![LR_Original Fold 3](./cv_plots/LR_Original_fold3.png) ![LR_Original Fold 4](./cv_plots/LR_Original_fold4.png)

![LR_Original Fold 5](./cv_plots/LR_Original_fold5.png) ![LR_Original Fold 6](./cv_plots/LR_Original_fold6.png)

### LR_Original - Precision, Recall, F1-score Table

| Fold | Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| 1 | 0.8697 | 0.9200 | 0.8942 |
| 2 | 0.8893 | 0.9494 | 0.9184 |
| 3 | 0.9143 | 0.9573 | 0.9353 |
| 4 | 0.8760 | 0.9339 | 0.9041 |
| 5 | 0.9200 | 0.9388 | 0.9293 |
| 6 | 0.8851 | 0.9467 | 0.9149 |

### LR_Original - AUC Scores

| Fold | AUC |
|------|-----|
| 1 | 0.8978 |
| 2 | 0.9211 |
| 3 | 0.9009 |
| 4 | 0.9030 |
| 5 | 0.8922 |
| 6 | 0.8871 |
| **Mean** | **0.9003** |
| **Std** | **0.0107** |

## XGB_Original - All Folds Confusion Matrices

![XGB_Original Fold 1](./cv_plots/XGB_Original_fold1.png) ![XGB_Original Fold 2](./cv_plots/XGB_Original_fold2.png)

![XGB_Original Fold 3](./cv_plots/XGB_Original_fold3.png) ![XGB_Original Fold 4](./cv_plots/XGB_Original_fold4.png)

![XGB_Original Fold 5](./cv_plots/XGB_Original_fold5.png) ![XGB_Original Fold 6](./cv_plots/XGB_Original_fold6.png)

### XGB_Original - Precision, Recall, F1-score Table

| Fold | Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| 1 | 0.8554 | 0.9467 | 0.8987 |
| 2 | 0.8876 | 0.9662 | 0.9253 |
| 3 | 0.9048 | 0.9744 | 0.9383 |
| 4 | 0.8555 | 0.9648 | 0.9068 |
| 5 | 0.9073 | 0.9592 | 0.9325 |
| 6 | 0.8704 | 0.9631 | 0.9144 |

### XGB_Original - AUC Scores

| Fold | AUC |
|------|-----|
| 1 | 0.9018 |
| 2 | 0.9274 |
| 3 | 0.9224 |
| 4 | 0.9098 |
| 5 | 0.9014 |
| 6 | 0.8942 |
| **Mean** | **0.9095** |
| **Std** | **0.0119** |

## LR_RFECV - All Folds Confusion Matrices

![LR_RFECV Fold 1](./cv_plots/LR_RFECV_fold1.png) ![LR_RFECV Fold 2](./cv_plots/LR_RFECV_fold2.png)

![LR_RFECV Fold 3](./cv_plots/LR_RFECV_fold3.png) ![LR_RFECV Fold 4](./cv_plots/LR_RFECV_fold4.png)

![LR_RFECV Fold 5](./cv_plots/LR_RFECV_fold5.png) ![LR_RFECV Fold 6](./cv_plots/LR_RFECV_fold6.png)

### LR_RFECV - Precision, Recall, F1-score Table

| Fold | Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| 1 | 0.8667 | 0.9244 | 0.8946 |
| 2 | 0.8893 | 0.9494 | 0.9184 |
| 3 | 0.9146 | 0.9615 | 0.9375 |
| 4 | 0.8724 | 0.9339 | 0.9021 |
| 5 | 0.9237 | 0.9388 | 0.9312 |
| 6 | 0.8817 | 0.9467 | 0.9130 |

### LR_RFECV - AUC Scores

| Fold | AUC |
|------|-----|
| 1 | 0.8984 |
| 2 | 0.9212 |
| 3 | 0.8994 |
| 4 | 0.9006 |
| 5 | 0.8933 |
| 6 | 0.8863 |
| **Mean** | **0.8999** |
| **Std** | **0.0107** |

## XGB_RFECV - All Folds Confusion Matrices

![XGB_RFECV Fold 1](./cv_plots/XGB_RFECV_fold1.png) ![XGB_RFECV Fold 2](./cv_plots/XGB_RFECV_fold2.png)

![XGB_RFECV Fold 3](./cv_plots/XGB_RFECV_fold3.png) ![XGB_RFECV Fold 4](./cv_plots/XGB_RFECV_fold4.png)

![XGB_RFECV Fold 5](./cv_plots/XGB_RFECV_fold5.png) ![XGB_RFECV Fold 6](./cv_plots/XGB_RFECV_fold6.png)

### XGB_RFECV - Precision, Recall, F1-score Table

| Fold | Precision | Recall | F1 Score |
|------|-----------|--------|----------|
| 1 | 0.8966 | 0.9244 | 0.9103 |
| 2 | 0.9194 | 0.9620 | 0.9402 |
| 3 | 0.9378 | 0.9658 | 0.9516 |
| 4 | 0.8975 | 0.9648 | 0.9299 |
| 5 | 0.9315 | 0.9429 | 0.9371 |
| 6 | 0.9098 | 0.9508 | 0.9299 |

### XGB_RFECV - AUC Scores

| Fold | AUC |
|------|-----|
| 1 | 0.9479 |
| 2 | 0.9581 |
| 3 | 0.9553 |
| 4 | 0.9486 |
| 5 | 0.9446 |
| 6 | 0.9347 |
| **Mean** | **0.9482** |
| **Std** | **0.0076** |

