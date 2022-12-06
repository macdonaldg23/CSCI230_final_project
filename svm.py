import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
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

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)


print('\n\n------------------------------------------ Evaluating SVM ----------------------------------------------- \n\n')

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
plot_confusion_matrix(svm, X_test, y_test, cmap=plt.cm.Blues)
plt.show()


print('\n\n------------------------------------------ SVM Metrics ----------------------------------------------- \n\n')

print('accuracy:  %0.6f' % accuracy_score(y_true=y_test, y_pred=y_pred))
print('precision: %0.6f' % precision_score(y_true=y_test, y_pred=y_pred))
print('recall:    %0.6f' % recall_score(y_true=y_test, y_pred=y_pred))
print('f1:        %0.6f' % f1_score(y_true=y_test, y_pred=y_pred))
print('ROC-AUC:   %0.6f' % roc_auc_score(y_test, y_pred))

print('\n\n')
print('SVM classification report:\n', classification_report(y_test, y_pred))

print('\n\n------------------------------------- Recursive Feature Elimination ------------------------------------- \n\n')

estimator = SVC(random_state=0, 
            kernel = params['kernel'], 
            C=params['C'], 
            degree=params['degree'],
            gamma=params['gamma'])

selector = RFE(estimator, n_features_to_select=0.5, step=1)
selector = selector.fit(X_train, y_train)

selector.fit(X_train, y_train)
y_pred_RFE = selector.predict(X_test)

RFE_confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_RFE)
print(RFE_confmat)
plot_confusion_matrix(selector, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

print('accuracy:  %0.6f' % accuracy_score(y_true=y_test, y_pred=y_pred_RFE))
print('precision: %0.6f' % precision_score(y_true=y_test, y_pred=y_pred_RFE))
print('recall:    %0.6f' % recall_score(y_true=y_test, y_pred=y_pred_RFE))
print('f1:        %0.6f' % f1_score(y_true=y_test, y_pred=y_pred_RFE))
print('ROC-AUC:   %0.6f' % roc_auc_score(y_test, y_pred_RFE))

print('\n\n')
print('SVM with RFE classification report:\n', classification_report(y_test, y_pred_RFE))