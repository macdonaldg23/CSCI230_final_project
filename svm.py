"""
Support Vector Machine
CSCI 230 Final Project
Grace MacDonald, Warren Seeds, Carson Cooley, Petra Ilic, Sarah Martin
"""
import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFE



print('\n\n====================================== Beginning Support Vector Machine ============================================ \n\n')

df = eda.df
X_train = eda.X_train_scaled
X_test = eda.X_test_scaled
y_train = eda.y_train
y_test = eda.y_test
X_val = eda.X_val_scaled
y_val = eda.y_val


param_grid = [{'C': [0.001,0.01,0.1,1,10],
                'kernel': ['linear','rbf'], 
                'degree':[1,3,6,9],
                'gamma': ['scale','auto']
               }]

gs = GridSearchCV(estimator=SVC(random_state=0), 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_val, y_val)
print(gs.best_score_)
print(gs.best_params_)
params = gs.best_params_


svm = SVC(random_state=0, 
            kernel = params['kernel'], 
            C=params['C'], 
            degree=params['degree'],
            gamma=params['gamma'])

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=123)

X_kf = np.concatenate((X_train, X_test))
y_kf = np.concatenate((y_train, y_test))

predicted_targets = np.array([])
actual_targets = np.array([])

accuracies = []
precisions = []
recalls = []
f1s = []
roc_aucs = []
for train_index, test_index in kf.split(X_kf):
    X_train_kf, X_test_kf = X_kf[train_index], X_kf[test_index]
    y_train_kf, y_test_kf = y_kf[train_index], y_kf[test_index]
    
    svm.fit(X_train_kf, y_train_kf)
    y_pred_kf = svm.predict(X_test_kf)
    
    predicted_targets = np.append(predicted_targets, y_pred_kf)
    actual_targets = np.append(actual_targets, y_test_kf)
    
    accuracies.append(accuracy_score(y_true=y_test_kf, y_pred=y_pred_kf))
    precisions.append(precision_score(y_true=y_test_kf, y_pred=y_pred_kf))
    recalls.append(recall_score(y_true=y_test_kf, y_pred=y_pred_kf))
    f1s.append(f1_score(y_true=y_test_kf, y_pred=y_pred_kf))
    roc_aucs.append(roc_auc_score(y_test_kf, y_pred_kf))

cm_kf = confusion_matrix(y_true=actual_targets, y_pred=predicted_targets)
cm_kf_display = ConfusionMatrixDisplay(cm_kf)
cm_kf_display.from_predictions(y_true=actual_targets, y_pred=predicted_targets, cmap=plt.cm.Blues)
plt.title("10-Fold Cross Validation Confusion Matrix")
plt.show()

print('\n\n------------------------------------ 10-Fold Cross Validation Metrics ----------------------------------------- \n\n')

print("Accuracy: ", round(sum(accuracies)/k, 6))
print("Precision:", round(sum(precisions)/k, 6))
print("Recall:   ", round(sum(recalls)/k, 6))
print("F1:       ", round(sum(f1s)/k, 6))
print("ROC-AUC:  ", round(sum(roc_aucs)/k, 6))

print('\n\n')
print('Classification report:\n', classification_report(actual_targets, predicted_targets))

print('\n\n------------------------------------- Recursive Feature Elimination ------------------------------------- \n\n')

estimator = SVC(random_state=0, 
            kernel = params['kernel'], 
            C=params['C'], 
            degree=params['degree'],
            gamma=params['gamma'])

selector = RFE(estimator, n_features_to_select=0.5, step=1)

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
