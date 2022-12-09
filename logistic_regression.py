"""
Logistic Regression
CSCI 230 Final Project
Grace MacDonald, Warren Seeds, Carson Cooley, Petra Ilic, Sarah Martin
"""

import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

print('\n\n====================================== Beginning Logistic Regression Classifier ============================================ \n\n')

df = eda.df
X_train = eda.X_train_scaled
X_test = eda.X_test_scaled
y_train = eda.y_train
y_test = eda.y_test
X_val = eda.X_val_scaled
y_val = eda.y_val

print(df)


param_grid = [{'penalty': ['l1', 'l2', 'elasticnet', 'none'],
               'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01], 
               'C': [0.01, 0.1, 1.0, 10.0, 100.0],
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
               }]

gs = GridSearchCV(estimator=LogisticRegression(random_state=0), 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_val, y_val)
print(gs.best_score_)
print(gs.best_params_)
params = gs.best_params_


estimator = LogisticRegression(random_state=0,
                               penalty = params['penalty'],
                               tol=params['tol'],
                               C=params['C'],
                               solver=params['solver'])

n_features = 5

rfe = RFE(estimator, n_features_to_select=n_features, step=1)

print('\n\n------------------------- Evaluating LRC with RFE (' + str(n_features) + ' features) ------------------------------ \n\n')

k = 10
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)

X_kf = np.concatenate((X_train, X_test))
y_kf = np.concatenate((y_train, y_test))

predicted_targets = np.array([])
actual_targets = np.array([])

accuracies = []
precisions = []
recalls = []
f1s = []
roc_aucs = []
for train_index, test_index in kf.split(X_kf,y_kf):
    X_train_kf, X_test_kf = X_kf[train_index], X_kf[test_index]
    y_train_kf, y_test_kf = y_kf[train_index], y_kf[test_index]
    
    rfe.fit(X_train_kf, y_train_kf)
    y_pred_kf = rfe.predict(X_test_kf)
    
    predicted_targets = np.append(predicted_targets, y_pred_kf)
    actual_targets = np.append(actual_targets, y_test_kf)
    
    accuracies.append(rfe.score(X_test_kf, y_test_kf))
    precisions.append(precision_score(y_true=y_test_kf, y_pred=y_pred_kf))
    recalls.append(recall_score(y_true=y_test_kf, y_pred=y_pred_kf))
    f1s.append(f1_score(y_true=y_test_kf, y_pred=y_pred_kf))
    roc_aucs.append(roc_auc_score(y_test_kf, y_pred_kf))

cm_kf = confusion_matrix(y_true=actual_targets, y_pred=predicted_targets)
cm_kf_display = ConfusionMatrixDisplay(cm_kf)
cm_kf_display.from_predictions(y_true=actual_targets, y_pred=predicted_targets, cmap=plt.cm.Blues)
plt.title("10-Fold Cross Validation Confusion Matrix")
plt.show()

print('\n\n------------------------------------ 10-Fold Cross Validation Metrics (RFE) ----------------------------------------- \n\n')

print("Accuracy: ", round(sum(accuracies)/k, 6))
print("Precision:", round(sum(precisions)/k, 6))
print("Recall:   ", round(sum(recalls)/k, 6))
print("F1:       ", round(sum(f1s)/k, 6))
print("ROC-AUC:  ", round(sum(roc_aucs)/k, 6))

print('\n\n')
print('Classification report:\n', classification_report(actual_targets, predicted_targets))



'''
rfe.fit(X_train, y_train)
estimator.fit(X_train, y_train)
y_pred_RFE = rfe.predict(X_test)
y_pred = estimator.predict(X_test)

print('\n\n------------------------------------------ RFE LRC Metrics (' + str(n_features) + ' features) ----------------------------------------------- \n\n')

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_RFE)
print(confmat)
plot_confusion_matrix(rfe, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

print('accuracy:  %0.6f' % accuracy_score(y_true=y_test, y_pred=y_pred_RFE))
print('precision: %0.6f' % precision_score(y_true=y_test, y_pred=y_pred_RFE))
print('recall:    %0.6f' % recall_score(y_true=y_test, y_pred=y_pred_RFE))
print('f1:        %0.6f' % f1_score(y_true=y_test, y_pred=y_pred_RFE))
print('ROC-AUC:   %0.6f' % roc_auc_score(y_test, y_pred_RFE))

print('\n\n')
print('LR classification report:\n', classification_report(y_test, y_pred_RFE))
'''
print('\n\n------------------------------------------ LRC Evaluation (No RFE or SBS) ----------------------------------------------- \n\n')
'''
confmat_RFE = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
plot_confusion_matrix(estimator, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

print('accuracy:  %0.6f' % accuracy_score(y_true=y_test, y_pred=y_pred))
print('precision: %0.6f' % precision_score(y_true=y_test, y_pred=y_pred))
print('recall:    %0.6f' % recall_score(y_true=y_test, y_pred=y_pred))
print('f1:        %0.6f' % f1_score(y_true=y_test, y_pred=y_pred))
print('ROC-AUC:   %0.6f' % roc_auc_score(y_test, y_pred))

print('\n\n')
print('LR classification report:\n', classification_report(y_test, y_pred))
'''

# predicted_targets_lrc = np.array([])
# actual_targets_lrc = np.array([])

# accuracies_lrc = []
# precisions_lrc = []
# recalls_lrc = []
# f1s_lrc = []
# roc_aucs_lrc = []
# for train_index, test_index in kf.split(X_kf):
#     X_train_kf, X_test_kf = X_kf[train_index], X_kf[test_index]
#     y_train_kf, y_test_kf = y_kf[train_index], y_kf[test_index]
    
#     estimator.fit(X_train_kf, y_train_kf)
#     y_pred_kf = rfe.predict(X_test_kf)
    
#     predicted_targets_lrc = np.append(predicted_targets, y_pred_kf)
#     actual_targets_lrc = np.append(actual_targets, y_test_kf)
    
#     accuracies_lrc.append(estimator.score(X_test_kf, y_test_kf))
#     precisions_lrc.append(precision_score(y_true=y_test_kf, y_pred=y_pred_kf))
#     recalls_lrc.append(recall_score(y_true=y_test_kf, y_pred=y_pred_kf))
#     f1s_lrc.append(f1_score(y_true=y_test_kf, y_pred=y_pred_kf))
#     roc_aucs_lrc.append(roc_auc_score(y_test_kf, y_pred_kf))

# cm_kf_lrc = confusion_matrix(y_true=actual_targets_lrc, y_pred=predicted_targets_lrc)
# cm_kf_display_lrc = ConfusionMatrixDisplay(cm_kf_lrc)
# cm_kf_display_lrc.from_predictions(y_true=actual_targets_lrc, y_pred=predicted_targets_lrc, cmap=plt.cm.Blues)
# plt.title("10-Fold Cross Validation Confusion Matrix")
# plt.show()

# print('\n\n------------------------------------ 10-Fold Cross Validation Metrics (no RFE) ----------------------------------------- \n\n')

# print("Accuracy: ", round(sum(accuracies_lrc)/k, 6))
# print("Precision:", round(sum(precisions_lrc)/k, 6))
# print("Recall:   ", round(sum(recalls_lrc)/k, 6))
# print("F1:       ", round(sum(f1s_lrc)/k, 6))
# print("ROC-AUC:  ", round(sum(roc_aucs_lrc)/k, 6))

# print('\n\n')
# print('Classification report:\n', classification_report(actual_targets_lrc, predicted_targets_lrc))

