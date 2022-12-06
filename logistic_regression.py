import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score
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

print('\n\n------------------------------------------ LRC Evaluation (No RFE or SBS) ----------------------------------------------- \n\n')

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

