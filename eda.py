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

print('\n==================Exploratory Data Analysis==================\n')


print('Importing the dataset...')
d = loadarff('1year.arff')
df1 = pd.DataFrame(d[0])

d = loadarff('2year.arff')
df2 = pd.DataFrame(d[0])

d = loadarff('3year.arff')
df3 = pd.DataFrame(d[0])

d = loadarff('4year.arff')
df4 = pd.DataFrame(d[0])

d = loadarff('5year.arff')
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


#breaks data into 4 groups so we can actually look at graphs
col1 = ['Attr1', 'Attr2','Attr3','Attr4','Attr5','Attr6','Attr7','Attr8','Attr9','Attr10','Attr11','Attr12','Attr13','Attr14','Attr15','Attr16','class']
col2 = ['Attr17', 'Attr18','Attr19','Attr20','Attr21','Attr22','Attr23','Attr24','Attr25','Attr26','Attr27','Attr28','Attr29','Attr30','Attr31','Attr32','class']
col3 = ['Attr33', 'Attr34','Attr35','Attr36','Attr37','Attr38','Attr39','Attr40','Attr41','Attr42','Attr43','Attr44','Attr45','Attr46','Attr47','Attr48','class']
col4 = ['Attr49', 'Attr50','Attr51','Attr52','Attr53','Attr54','Attr55','Attr56','Attr57','Attr58','Attr59','Attr60','Attr61','Attr62','Attr63','Attr64','class']


# Violin plot
features = df.iloc[:,0:63]
plt.figure(figsize=(10,40))
j = 0
for i in features:
    plt.subplot(21,3,j+1)
    sns.violinplot(x=df["class"],y=df[i],palette=["red","green"])
    plt.title(i)
    plt.axhline(df[i].mean(),linestyle = "dashed", label ="Mean value = " + str(round(df[i].mean(), 2)))
    plt.legend(loc="best")
    j = j + 1


''' can not do Pearson's R until we deal with null values'''
# cm = np.corrcoef(df[col1].values.T)
# hm = heatmap(cm,row_names=col1,column_names=col1, figsize=(8,8))
# plt.show()


# cm = np.corrcoef(df[col2].values.T)
# hm = heatmap(cm,row_names=col2,column_names=col2, figsize=(8,8))
# plt.show()


# cm = np.corrcoef(df[col3].values.T)
# hm = heatmap(cm,row_names=col3,column_names=col3, figsize=(8,8))
# plt.show()


# cm = np.corrcoef(df[col4].values.T)
# hm = heatmap(cm,row_names=col4,column_names=col4, figsize=(8,8))
# plt.show()