"""
Decision Tree Classifier
CSCI 230 Final Project
Grace MacDonald, Warren Seeds, Carson Cooley, Petra Ilic, Sarah Martin
"""
import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, validation_curve, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, export_graphviz, plot_tree
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
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
for train_index, test_index in kf.split(X_kf, y_kf):
    X_train_kf, X_test_kf = X_kf[train_index], X_kf[test_index]
    y_train_kf, y_test_kf = y_kf[train_index], y_kf[test_index]
    
    dtc.fit(X_train_kf, y_train_kf)
    y_pred_kf = dtc.predict(X_test_kf)
    
    predicted_targets = np.append(predicted_targets, y_pred_kf)
    actual_targets = np.append(actual_targets, y_test_kf)
    
    accuracies.append(dtc.score(X_test_kf, y_test_kf))
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

# print('\n\n----------------------------- Printing Graphical Representation of DTC ------------------------------------- \n\n')


# plot_tree(dtc, fontsize=5, feature_names=eda.X_train.columns)
# plt.tight_layout()
# plt.show()

# print('\n\n------------------------------------------ Feature Importance ----------------------------------------------- \n\n')


# feature_importance = pd.DataFrame(dtc.feature_importances_,
#                                 index = X_train.columns,
#                                 columns=['importance']).sort_values('importance', 
#                                                                     ascending=False)

# print(feature_importance)

# feature_importance = feature_importance[feature_importance['importance'] != 0]

# ax = feature_importance.plot.bar(y='importance',rot=0)
# plt.show()


print('\n\n------------------------------------- Recursive Feature Elimination ------------------------------------- \n\n')

estimator = DecisionTreeClassifier(random_state=0, criterion=params['criterion'], max_depth=params['max_depth'], 
                            max_features=params['max_features'],  
                            min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'], 
                            min_weight_fraction_leaf=params['min_weight_fraction_leaf'], splitter=params['splitter'])

selector = RFE(estimator, n_features_to_select=0.3, step=1)

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