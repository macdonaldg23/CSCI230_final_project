import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score
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

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
plot_confusion_matrix(rfc, X_test, y_test, cmap=plt.cm.Blues)
plt.show()


print('\n\n------------------------------------------ RFC Metrics ----------------------------------------------- \n\n')

print('accuracy:  %0.6f' % accuracy_score(y_true=y_test, y_pred=y_pred))
print('precision: %0.6f' % precision_score(y_true=y_test, y_pred=y_pred))
print('recall:    %0.6f' % recall_score(y_true=y_test, y_pred=y_pred))
print('f1:        %0.6f' % f1_score(y_true=y_test, y_pred=y_pred))
print('ROC-AUC:   %0.6f' % roc_auc_score(y_test, y_pred))

print('\n\n')
print('RF classification report:\n', classification_report(y_test, y_pred))

print('\n\n------------------------------------------ Feature Importance ----------------------------------------------- \n\n')

print(rfc.feature_importances_)

feature_importance = pd.DataFrame(rfc.feature_importances_,
                                index = X_train.columns,
                                columns=['importance']).sort_values('importance', 
                                                                    ascending=False)

print(feature_importance)


print('\n\n------------------------------------- Recursive Feature Elimination ------------------------------------- \n\n')

estimator = RandomForestClassifier(random_state=0, 
                                n_estimators = params['n_estimators'], 
                                criterion=params['criterion'], 
                                max_depth=params['max_depth'],
                                max_features=params['max_features'])

selector = RFE(estimator, n_features_to_select=0.25, step=1)
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
print('RF classification report:\n', classification_report(y_test, y_pred_RFE))