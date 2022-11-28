"""
CSCI230 Final Project
Petra Ilic, Grace MacDonald, Carson Cooley, Sarah Martin, Warren Seeds
Exploratory Data Analysis
"""
import pandas as pd 
import numpy as np 
from scipy.io.arff import loadarff
from mlxtend.plotting import scatterplotmatrix, heatmap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.model_selection import train_test_split



print('\n==================Exploratory Data Analysis==================\n')


print('Importing the dataset...')
d = loadarff('data/1year.arff')
df1 = pd.DataFrame(d[0])

d = loadarff('data/2year.arff')
df2 = pd.DataFrame(d[0])

d = loadarff('data/3year.arff')
df3 = pd.DataFrame(d[0])

d = loadarff('data/4year.arff')
df4 = pd.DataFrame(d[0])

d = loadarff('data/5year.arff')
df5 = pd.DataFrame(d[0])

df = df1 + df2 + df3 + df4 + df5

print(df)


print('\n==================Columns and their datatypes==================\n')
print(df.dtypes)

print('\n==================Checking for null values==================\n')

print(df.isnull().sum())

''' 
4593 NaN values for class variable -- drop right away because we do not want
                                      to impute target
    leaves us with 5910 rows
'''

print('\nDropping null values from class variable...\n')
df = df[df['class'].notna()]


# converting to int, can only be done after dropping na values
df = df.astype({'class': 'int64'})


print('\n==================Columns and their datatypes, updated==================\n')
print(df.dtypes)


print('\n==================Checking for null values==================\n')
print(df.isnull().sum())

print('\n==================Checking for rows with all null values==================\n')
print(df.isna().all(axis=1).sum())

print('\n==================Value Counts for Class Column==================\n')
print(df['class'].value_counts())


'''will need to do some imputation --> without it there are only 111 rows left '''

print('df size: ', df.shape)


print('\n==================Descriptive stats on Data:==================\n')
print(df.describe())


#breaks data into 4 groups so we can actually look at graphs
col1 = ['Attr1', 'Attr2','Attr3','Attr4','Attr5','Attr6','Attr7','Attr8','Attr9','Attr10','Attr11','Attr12','Attr13','Attr14','Attr15','Attr16']
col2 = ['Attr17', 'Attr18','Attr19','Attr20','Attr21','Attr22','Attr23','Attr24','Attr25','Attr26','Attr27','Attr28','Attr29','Attr30','Attr31','Attr32']
col3 = ['Attr33', 'Attr34','Attr35','Attr36','Attr37','Attr38','Attr39','Attr40','Attr41','Attr42','Attr43','Attr44','Attr45','Attr46','Attr47','Attr48']
col4 = ['Attr49', 'Attr50','Attr51','Attr52','Attr53','Attr54','Attr55','Attr56','Attr57','Attr58','Attr59','Attr60','Attr61','Attr62','Attr63','Attr64']

""" # Violin plot
features = df.iloc[:,0:63]
plt.figure(figsize=(10,40))
j = 0
for i in features:
    plt.subplot(21,3,j+1)
    sns.violinplot(x=df["class"],y=df[i],palette=["red","green"])
    plt.title(i)
    plt.axhline(df[i].mean(),linestyle = "dashed", label ="Mean value = " + str(round(df[i].mean(), 2)))
    plt.legend(loc="best")
    j = j + 1 """


print('\n\nImputing with mean...')
df = df.fillna(df.mean())


# sorted_values = df['class'].value_counts().sort_values(ascending=False)
# sorted_columns = df['class'].value_counts().index.tolist()

# min_len = sorted_values.iloc[1]
# print('min_len', min_len)

# for i in range(0, len(sorted_values)-1):
#     maj_len = sorted_values.iloc[i]
#     frac = ((maj_len - min_len) / maj_len)
#     if frac > 0:
#         indexes = df[df['class']==sorted_columns[i]].sample(frac=frac, random_state=0).index
#         df = df.drop(indexes)

# print('df size: ', df.shape)




''' can not do Pearson's R until we deal with null values'''
# cm = np.corrcoef(df[col1 + ['class']].values.T)
# hm = heatmap(cm,row_names=col1 + ['class'],column_names=col1 + ['class'], figsize=(8,8))
# plt.show()


# cm = np.corrcoef(df[col2 + ['class']].values.T)
# hm = heatmap(cm,row_names=col2 + ['class'],column_names=col2 + ['class'], figsize=(8,8))
# plt.show()


# cm = np.corrcoef(df[col3 + ['class']].values.T)
# hm = heatmap(cm,row_names=col3 + ['class'],column_names=col3 + ['class'], figsize=(8,8))
# plt.show()


# cm = np.corrcoef(df[col4 + ['class']].values.T)
# hm = heatmap(cm,row_names=col4 + ['class'],column_names=col4 + ['class'], figsize=(8,8))
# plt.show()

# '''Get k best features based on ANOVA-F statistic '''
# X_f = df[col1+col2+col3+col4]
# y_f = df['class']

# select = SelectKBest(score_func=f_classif,k=20)
# new_feats = select.fit_transform(X_f,y_f)

# filter = select.get_support()

# print(filter)

# print('New shape:', new_feats.shape)

# print(X_f.columns[filter])


print(df['class'].value_counts())

# Note that downsampling must be done before splitting the data
sorted_values = df['class'].value_counts().sort_values(ascending=False)
sorted_columns = df['class'].value_counts().index.tolist()

print(sorted_values)
print(sorted_columns)

#decrease majority by half
#min_len = (sorted_values.iloc[0] + sorted_values.iloc[-1]) // 2


#make majority same number as minority
min_len = sorted_values.iloc[1]
print('min_len', min_len)

for i in range(0, len(sorted_values)-1):
    maj_len = sorted_values.iloc[i]
    frac = ((maj_len - min_len) / maj_len)
    if frac > 0:
        indexes = df[df['class']==sorted_columns[i]].sample(frac=frac, random_state=0).index
        df = df.drop(indexes)

print(df['class'].value_counts())
print(df.shape)

X = df[col1+col2+col3+col4]
print(X)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)



# f_stat = f_classif(X,y)

# anova_f_res = pd.DataFrame(f_stat,index=['f_statsitc','p_value'],columns=col1+col2+col3+col4)


# print('\n==================ANOVA-F==================\n')

# print(anova_f_res)
# print('\n\n')

# f_val = pd.Series(f_stat[0],index=col1+col2+col3+col4)s
# print(f_val.sort_values())