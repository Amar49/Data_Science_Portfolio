import pandas as pd
import numpy as np

#Task: Given attributes about the person, predict income is >50K and <50K

df = pd.read_csv('adult.csv')
print(df.head(5))

print(df['income'].value_counts())

#Remove spaces in a cell
df['income'] = [x.strip().replace(' ', '') for x in df['income']]

df['native_country'] = [x.strip().replace(' ','') for x in df['native_country']]

df['income'] = [0 if x == '<=50K' else 1 for x in df['income']]

print(df['income'].value_counts())
 

X = df.drop('income',1)
Y = df['income']

print(X.head(5))
print(Y.head(5))

#Label encoding and creating dummy features using get_dummies
print(pd.get_dummies(X['education']).head(5))

# Decide which categorical variables you want to use in model
for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_cat = len(X[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))

# Although, 'native_country' has a lot of unique categories, most categories only have a few observations
print(X['native_country'].value_counts().sort_values(ascending=False).head(10))

# In this case, bucket low frequecy categories as "Other"
X['native_country'] = ['United-States' if x == 'United-States' else 'Other' for x in X['native_country']]

print(X['native_country'].value_counts().sort_values(ascending=False))

# Create a list of features to dummy
todummy_list = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df

X = dummy_df(X, todummy_list)
print(X.head(5))

# How much of your data is missing?
print(X.isnull().sum().sort_values(ascending=False).head())

# Impute missing values using Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)

# Now check again to see if you still have missing data
print(X.isnull().sum().sort_values(ascending=False).head())


# Impute missing values using Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)

# Now check again to see if you still have missing data
print(X.isnull().sum().sort_values(ascending=False).head())

# Impute missing values using Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)

# Now check again to see if you still have missing data
print(X.isnull().sum().sort_values(ascending=False).head())

def find_outliers_tukey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1 
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values

tukey_indices, tukey_values = find_outliers_tukey(X['age'])
print(np.sort(tukey_values))

from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate

def find_outliers_kde(x):
    x_scaled = scale(list(map(float, x)))
    kde = KDEUnivariate(x_scaled)
    kde.fit(bw="scott", fft=True)
    pred = kde.evaluate(x_scaled)
    
    n = sum(pred < 0.05)
    outlier_ind = np.asarray(pred).argsort()[:n]
    outlier_value = np.asarray(x)[outlier_ind]

    return outlier_ind, outlier_value

kde_indices, kde_values = find_outliers_kde(X['age'])
print(np.sort(kde_values))

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)

print(df.shape)
print(X.shape)

import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]

print(colnames_selected)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def find_model_perf(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(accuracy_score(y_test, predictions))

find_model_perf(X_train, y_train, X_test, y_test)
