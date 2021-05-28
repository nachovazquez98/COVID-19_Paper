#%%
import os, sys
import pandas as pd
import numpy as np
import joblib
from gridsearchcv import Gridsearchcv
from sklearn.model_selection import train_test_split
import pathlib
from sklearn.metrics import classification_report, confusion_matrix
#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from pipelinehelper import PipelineHelper
from sklearn.inspection import permutation_importance
#%%
from sklearn.datasets import load_breast_cancer 
cancer = load_breast_cancer() 
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names']) 
df['target'] = cancer['target']
df.head()
file_name = 'breast_cancer'
#%%#separar datos
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify=y, shuffle=True)
grid, df_grid, grid_report= Gridsearchcv(X_train, X_test, y_train, y_test, randomizedsearch=True)
#%%#guarda el modelo y su reporte
joblib.dump(grid, 'models/'+file_name+'_grid.pkl', compress = 1)
df_grid.to_csv('models/'+file_name+'_df_grid.csv', index=True)
grid_report.to_csv('models/'+file_name+'_grid_report.csv', index=True)
#%%validation curve
#%%importa el modelo y su rendimiento
grid=joblib.load('models/'+file_name+'_grid.pkl')
grid_report=pd.read_csv('models/'+file_name+'_grid_report.csv',index_col=0)

test_scores = grid.cv_results_['mean_test_ba']
train_scores = grid.cv_results_['mean_train_ba'] 

plt.plot(test_scores, label='test')
plt.plot(train_scores, label='train')
plt.legend(loc='best')
plt.show()
#%%extract best parameter
#grid_report.iloc[0,:]
#grid_report.iloc[0,:][0][1]
'''
#%%Pipeline without gridsearchcv
num_transformer=Pipeline(steps=[
    ('scaler',MinMaxScaler())])

preprocessor=ColumnTransformer(
    remainder='passthrough',
    transformers=[('num',num_transformer,['EDAD'])])

svc_model=Pipeline([
    #('preprocessor',preprocessor),
    ('svc',SVC(probability=True))])

#svc_model.set_params(**grid.best_params_)
#svc_model.set_params(**{'SupVM__C':500,'SupVM__gamma':1})
#svc_model.set_params(**{'SupVM__C': 75, 'SupVM__gamma': 5})

svc_model.set_params(**{'svc__C': 1, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'})

print("Model params: ",svc_model.get_params("model"))
svc_model.fit(X_train,y_train)
y_pred=svc_model.predict(X_test)
print(classification_report(y_test,y_pred))
'''
#%%
#GRAFICAR FEATURE IMPORTANCE