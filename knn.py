"""
K- Nearest Neighbors
CSCI 230 Final Project
Grace MacDonald, Warren Seeds, Carson Cooley, Petra Ilic, Sarah Martin
"""

import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

print('\n\n================================== Beginning K-Nearest Neighbors Classifier ======================================= \n\n')

df = eda.df
X_train = eda.X_train_scaled
X_test = eda.X_test_scaled
y_train = eda.y_train
y_test = eda.y_test
X_val = eda.X_val_scaled
y_val = eda.y_val

X_kf = np.concatenate((X_train, X_test))
y_kf = np.concatenate((y_train, y_test))

param_grid = [{'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10],
               'weights': ['uniform','distance'], 
               'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
               'p': [1,2]
               }]

gs = GridSearchCV(estimator=KNeighborsClassifier(), 
                  param_grid=param_grid, 
                  scoring='recall', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_val, y_val)
print(gs.best_score_)
print(gs.best_params_)
params = gs.best_params_


knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'], 
                           weights=params['weights'], 
                           algorithm=params['algorithm'], 
                           p=params['p'])

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('\n\n------------------------------------------ Evaluating KNN ----------------------------------------------- \n\n')

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=123)

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
    
    knn.fit(X_train_kf, y_train_kf)
    y_pred_kf = knn.predict(X_test_kf)
    
    predicted_targets = np.append(predicted_targets, y_pred_kf)
    actual_targets = np.append(actual_targets, y_test_kf)
    
    accuracies.append(knn.score(X_test_kf, y_test_kf))
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

print('\n\n------------------------------------- Principal Component Analysis ------------------------------------- \n\n')

components = [x for x in range(1, 63)]

best_comp = 0
best_score = 0
for comp in components:

    pca = PCA(comp)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    knn.fit(X_train_pca, y_train)
    y_pred_pca = knn.predict(X_test_pca)
    
    recall = recall_score(y_true=y_test, y_pred=y_pred_pca)
    
    if recall > best_score:
        best_score = recall
        best_comp = comp

pca = PCA(best_comp)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_kf_pca = np.concatenate((X_train_pca, X_test_pca))

predicted_targets_pca = np.array([])
actual_targets_pca = np.array([])

pca_accuracies = []
pca_precisions = []
pca_recalls = []
pca_f1s = []
pca_roc_aucs = []

for train_index, test_index in kf.split(X_kf_pca):
    X_train_kf, X_test_kf = X_kf_pca[train_index], X_kf_pca[test_index]
    y_train_kf, y_test_kf = y_kf[train_index], y_kf[test_index]
    
    knn.fit(X_train_kf, y_train_kf)
    y_pred_kf = knn.predict(X_test_kf)
    
    predicted_targets_pca = np.append(predicted_targets_pca, y_pred_kf)
    actual_targets_pca = np.append(actual_targets_pca, y_test_kf)
    
    pca_accuracies.append(knn.score(X_test_kf, y_test_kf))
    pca_precisions.append(precision_score(y_true=y_test_kf, y_pred=y_pred_kf))
    pca_recalls.append(recall_score(y_true=y_test_kf, y_pred=y_pred_kf))
    pca_f1s.append(f1_score(y_true=y_test_kf, y_pred=y_pred_kf))
    pca_roc_aucs.append(roc_auc_score(y_test_kf, y_pred_kf))

cm_kf = confusion_matrix(y_true=actual_targets_pca, y_pred=predicted_targets_pca)
cm_kf_display = ConfusionMatrixDisplay(cm_kf)
cm_kf_display.from_predictions(y_true=actual_targets_pca, y_pred=predicted_targets_pca, cmap=plt.cm.Blues)
plt.title("10-Fold Cross Validation Confusion Matrix")
plt.show()

print("Accuracy: ", round(sum(pca_accuracies)/k, 6))
print("Precision:", round(sum(pca_precisions)/k, 6))
print("Recall:   ", round(sum(pca_recalls)/k, 6))
print("F1:       ", round(sum(pca_f1s)/k, 6))
print("ROC-AUC:  ", round(sum(pca_roc_aucs)/k, 6))

print('\n\n')
print('Classification report:\n', classification_report(actual_targets_pca, predicted_targets_pca))