import numpy as np
import pandas as pd 
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.tree import DecisionTreeClassifier


df = pd.read_excel('C:\\Personal\\EndtoEndProject\\emailfabrication\\inputdata.xlsx', engine='openpyxl')

column_names = []
for i in df.columns:
    column_names.append(i)

for x in column_names:
    df[x] = df[x].astype('category')

target = df['email']

label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'category':
        df[col] = label_encoder.fit_transform(df[col])

X = df.drop(['email'], axis=1)
y = target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

# Model Specification
models = {
    "Random Forest": RandomForestClassifier()
    }

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print(list(models.keys())[i])
    print('Train metric scores')
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(train_accuracy)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    print(train_precision)
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    print(train_recall)
    
    print('===================')
    
    print('Test metric scores')
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(test_accuracy)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    print(test_precision)
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    print(test_recall)
    
    print('\n--------------------')
    

# Hyper-parameter Tuning
"""
rf_params = {'max_depth':[None, 5, 8, 10, 15, 20],
             'max_features':[5, 8, 7, 'auto', 'sqrt'],
             'min_samples_split':[2, 5, 8, 10, 15],
             'n_estimators':[100, 200, 500, 1000],
             'criterion':['gini', 'entropy']
    }

randomcv_models = [
                ('RF', RandomForestClassifier(), rf_params)
                ]

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator = model, 
                                param_distributions = params,
                                n_iter = 100,
                                cv = 3,
                                n_jobs = -1)
    
    random.fit(X_train, y_train)
    param = ', '.join([f'{k}={v}' for k, v in random.best_params_.items()])
    print(random.best_params_)
"""
    
# n_estimators = 200, min_samples_split = 2, max_features ='auto', max_depth = 10, criterion = 'entropy'
# n_estimators = 1000, min_samples_split = 2, max_features = 5, max_depth = 15, criterion = 'gini'
# 
      
# Model Application
"""
rfm = RandomForestClassifier(n_estimators = 1000, min_samples_split = 2, max_features = 5, max_depth = 15, criterion = 'gini')
model_rf = rfm.fit(X_train, y_train)

y_train_pred = model_rf.predict(X_train)
y_test_pred = model_rf.predict(X_test)

print('Train metric scores')
train_accuracy = accuracy_score(y_train, y_train_pred)
print(train_accuracy)
train_precision = precision_score(y_train, y_train_pred, average='weighted')
print(train_precision)
train_recall = recall_score(y_train, y_train_pred, average='weighted')
print(train_recall)

print('Test metric scores')
test_accuracy = accuracy_score(y_test, y_test_pred)
print(test_accuracy)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
print(test_precision)
test_recall = recall_score(y_test, y_test_pred, average='weighted')
print(test_recall)
"""

pickle.dump(model, open('emailfab.pkl','wb'))
pickled_model = pickle.load(open('emailfab.pkl','rb'))



