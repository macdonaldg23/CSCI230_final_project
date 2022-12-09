"""
Random Forest Classifier
CSCI 230 Final Project
Grace MacDonald, Warren Seeds, Carson Cooley, Petra Ilic, Sarah Martin
"""

import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

print('\n\n====================================== Beginning Random Forest Classifier ============================================ \n\n')

df = eda.df
X_train = eda.X_train
X_test = eda.X_test
y_train = eda.y_train
y_test = eda.y_test
X_val = eda.X_val
y_val = eda.y_val

param_grid = [{'n_estimators': [50,100,200,300],
               'criterion': ['gini', 'entropy', 'log_loss'], 
               'max_depth': [5,10,20,50,100,200],
               'max_features': [None,'sqrt', 'log2']
               }]

gs = GridSearchCV(estimator=RandomForestClassifier(random_state=0), 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_val, y_val)
print(gs.best_score_)
print(gs.best_params_)
params = gs.best_params_


rfc = RandomForestClassifier(random_state=0, 
                                n_estimators = params['n_estimators'], 
                                criterion=params['criterion'], 
                                max_depth=params['max_depth'],
                                max_features=params['max_features'])

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)


print('\n\n------------------------------------------ Evaluating RFC ----------------------------------------------- \n\n')
k = 10
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=123)

X_kf = np.concatenate((X_train, X_test))
y_kf = np.concatenate((y_train, y_test))

# predicted_targets = np.array([])
# actual_targets = np.array([])

# accuracies = []
# precisions = []
# recalls = []
# f1s = []
# roc_aucs = []
# for train_index, test_index in kf.split(X_kf, y_kf):
#     X_train_kf, X_test_kf = X_kf[train_index], X_kf[test_index]
#     y_train_kf, y_test_kf = y_kf[train_index], y_kf[test_index]
    
#     rfc.fit(X_train_kf, y_train_kf)
#     y_pred_kf = rfc.predict(X_test_kf)
    
#     predicted_targets = np.append(predicted_targets, y_pred_kf)
#     actual_targets = np.append(actual_targets, y_test_kf)
    
#     accuracies.append(accuracy_score(y_true=y_test_kf, y_pred=y_pred_kf))
#     precisions.append(precision_score(y_true=y_test_kf, y_pred=y_pred_kf))
#     recalls.append(recall_score(y_true=y_test_kf, y_pred=y_pred_kf))
#     f1s.append(f1_score(y_true=y_test_kf, y_pred=y_pred_kf))
#     roc_aucs.append(roc_auc_score(y_test_kf, y_pred_kf))

# cm_kf = confusion_matrix(y_true=actual_targets, y_pred=predicted_targets)
# cm_kf_display = ConfusionMatrixDisplay(cm_kf)
# cm_kf_display.from_predictions(y_true=actual_targets, y_pred=predicted_targets, cmap=plt.cm.Blues)
# plt.title("10-Fold Cross Validation Confusion Matrix")
# plt.show()

# print('\n\n------------------------------------ 10-Fold Cross Validation Metrics ----------------------------------------- \n\n')

# print("Accuracy: ", round(sum(accuracies)/k, 6))
# print("Precision:", round(sum(precisions)/k, 6))
# print("Recall:   ", round(sum(recalls)/k, 6))
# print("F1:       ", round(sum(f1s)/k, 6))
# print("ROC-AUC:  ", round(sum(roc_aucs)/k, 6))

# print('\n\n')
# print('Classification report:\n', classification_report(actual_targets, predicted_targets))


print('\n\n------------------------------------- Recursive Feature Elimination ------------------------------------- \n\n')

estimator = RandomForestClassifier(random_state=0, 
                                   n_estimators = params['n_estimators'], 
                                   criterion=params['criterion'], 
                                   max_depth=params['max_depth'],
                                   max_features=params['max_features'])

selector = RFE(estimator, n_features_to_select=0.25, step=1)

predicted_targets_RFE = np.array([])
actual_targets_RFE = np.array([])

accuracies_RFE = []
precisions_RFE = []
recalls_RFE = []
f1s_RFE = []
roc_aucs_RFE = []
for train_index, test_index in kf.split(X_kf, y_kf):
    X_train_kf, X_test_kf = X_kf[train_index], X_kf[test_index]
    y_train_kf, y_test_kf = y_kf[train_index], y_kf[test_index]
    
    selector.fit(X_train_kf, y_train_kf)
    y_pred_kf = selector.predict(X_test_kf)
    
    predicted_targets_RFE = np.append(predicted_targets_RFE, y_pred_kf)
    actual_targets_RFE = np.append(actual_targets_RFE, y_test_kf)
    
    accuracies_RFE.append(round(accuracy_score(y_true=y_test_kf, y_pred=y_pred_kf), 6))
    precisions_RFE.append(round(precision_score(y_true=y_test_kf, y_pred=y_pred_kf), 6))
    recalls_RFE.append(round(recall_score(y_true=y_test_kf, y_pred=y_pred_kf), 6))
    f1s_RFE.append(round(f1_score(y_true=y_test_kf, y_pred=y_pred_kf), 6))
    roc_aucs_RFE.append(round(roc_auc_score(y_test_kf, y_pred_kf), 6))

cm_kf_RFE = confusion_matrix(y_true=actual_targets_RFE, y_pred=predicted_targets_RFE)
cm_kf_display_RFE = ConfusionMatrixDisplay(cm_kf_RFE)
cm_kf_display_RFE.from_predictions(y_true=actual_targets_RFE, y_pred=predicted_targets_RFE, cmap=plt.cm.Blues)
plt.title("10-Fold Cross Validation Confusion Matrix")
plt.show()

print('\n\n------------------------------------ 10-Fold Cross Validation Metrics with RFE ----------------------------------------- \n\n')

print("Accuracy: ", round(sum(accuracies_RFE)/len(accuracies_RFE), 6))
print("Precision:", round(sum(precisions_RFE)/len(precisions_RFE), 6))
print("Recall:   ", round(sum(recalls_RFE)/len(recalls_RFE), 6))
print("F1:       ", round(sum(f1s_RFE)/len(f1s_RFE), 6))
print("ROC-AUC:  ", round(sum(roc_aucs_RFE)/len(roc_aucs_RFE), 6))

print("\n\nAccuracy: ", accuracies_RFE)
print("Precision:", precisions_RFE)
print("Recall:   ", recalls_RFE)
print("F1:       ", f1s_RFE)
print("ROC-AUC:  ", roc_aucs_RFE)

print('\n\n')
print('Classification report:\n', classification_report(actual_targets_RFE, predicted_targets_RFE))

# print('\n\n------------------------------------ Non-10-Fold Cross Validation Metrics ----------------------------------------- \n\n')
# selector.fit(X_train, y_train)
# y_pred_RFE = selector.predict(X_test)

# RFE_confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_RFE)
# print(RFE_confmat)
# plot_confusion_matrix(selector, X_test, y_test, cmap=plt.cm.Blues)
# plt.show()

# print('accuracy:  %0.6f' % accuracy_score(y_true=y_test, y_pred=y_pred_RFE))
# print('precision: %0.6f' % precision_score(y_true=y_test, y_pred=y_pred_RFE))
# print('recall:    %0.6f' % recall_score(y_true=y_test, y_pred=y_pred_RFE))
# print('f1:        %0.6f' % f1_score(y_true=y_test, y_pred=y_pred_RFE))
# print('ROC-AUC:   %0.6f' % roc_auc_score(y_test, y_pred_RFE))

# print('\n\n')
# print('RF classification report:\n', classification_report(y_test, y_pred_RFE))

# print('\n\n------------------------------------------ Feature Importance ----------------------------------------------- \n\n')

# print(rfc.feature_importances_)

# feature_importance = pd.DataFrame(rfc.feature_importances_,
#                                 index = X_train.columns,
#                                 columns=['importance']).sort_values('importance', 
#                                                                     ascending=False)

# print(feature_importance)
