import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, export_graphviz, plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import RFE


print('\n\n====================================== Beginning Decision Tree Classifier ============================================ \n\n')
df = eda.df
X_train = eda.X_train
X_test = eda.X_test
y_train = eda.y_train
y_test = eda.y_test
X_val = eda.X_val
y_val = eda.y_val


print('\n\n--------------------------- Addressing over and underfitting with validation curves ------------------------\n\n')

param_range = [1, 5, 10, 50, 100, 250, 500, 1000]
train_scores, test_scores = validation_curve(
                estimator=DecisionTreeClassifier(random_state=0), 
                X=X_train, 
                y=y_train, 
                param_name='max_depth', 
                param_range=param_range,
                cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='Training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')
plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='Validation accuracy')
plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter max_depth')
plt.ylabel('Accuracy')
plt.ylim([0.2, 1.03])
plt.tight_layout()
plt.show()

print('\n\n------------------------------------------- DTC Grid Search ------------------------------------------------ \n\n')

param_grid = [{'criterion': ['gini', 'entropy', 'log_loss'], 
               'max_depth': [3,5,10,20,50,100,200],
               'min_samples_split': [2,6,10], 
               'splitter': ['best', 'random'],
               'min_samples_leaf': [1,5,10],
               'min_weight_fraction_leaf': [0,0.2,0.35,0.5],
               'max_features': ['sqrt', 'log2']
               }]

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X_val, y_val)
print(gs.best_score_)
print(gs.best_params_)
params = gs.best_params_

dtc = DecisionTreeClassifier(random_state=0, criterion=params['criterion'], max_depth=params['max_depth'], 
                            max_features=params['max_features'],  
                            min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'], 
                            min_weight_fraction_leaf=params['min_weight_fraction_leaf'], splitter=params['splitter'])
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)


'''
print('\n\n------------------------------------------ Importance of Each Feature ----------------------------------------------- \n\n')
feature_importances = pd.DataFrame(dtc.feature_importances_,
                                index = X_train.columns)

print(feature_importances)
'''

print('\n\n------------------------------------------ Evaluating DTC ----------------------------------------------- \n\n')

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
plot_confusion_matrix(dtc, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

print('\n\n------------------------------------------ DTC Metrics ----------------------------------------------- \n\n')


print('accuracy:  %0.6f' % accuracy_score(y_true=y_test, y_pred=y_pred))
print('precision: %0.6f' % precision_score(y_true=y_test, y_pred=y_pred))
print('recall:    %0.6f' % recall_score(y_true=y_test, y_pred=y_pred))
print('f1:        %0.6f' % f1_score(y_true=y_test, y_pred=y_pred))
print('ROC-AUC:   %0.6f' % roc_auc_score(y_test, y_pred))

print('\n\n')
print('DT classification report:\n', classification_report(y_test, y_pred))


# print('\n\n----------------------------- Printing Graphical Representation of DTC ------------------------------------- \n\n')


# plot_tree(dtc, fontsize=5, feature_names=eda.X_train.columns)
# plt.tight_layout()
# plt.show()

print('\n\n------------------------------------------ Feature Importance ----------------------------------------------- \n\n')


feature_importance = pd.DataFrame(dtc.feature_importances_,
                                index = X_train.columns,
                                columns=['importance']).sort_values('importance', 
                                                                    ascending=False)

print(feature_importance)

feature_importance = feature_importance[feature_importance['importance'] != 0]

ax = feature_importance.plot.bar(y='importance',rot=0)
plt.show()