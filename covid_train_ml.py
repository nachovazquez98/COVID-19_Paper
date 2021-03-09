#%%
'''
robust versions of logistic regression
support vector machines
random forests
gradient boosted decision trees
https://www.kdnuggets.com/2020/06/simplifying-mixed-feature-type-preprocessing-scikit-learn-pipelines.html
https://www.kaggle.com/kritidoneria/explainable-ai-eli5-lime-and-shap
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html
https://kiwidamien.github.io/introducing-the-column-transformer.html
https://medium.com/analytics-vidhya/shap-part-2-kernel-shap-3c11e7a971b1
'''

import os, sys
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV 
import joblib
sns.set(color_codes=True)
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
#%%abrir csv
path = "/home/nacho/Documents/coronavirus/COVID-19_Paper/"
os.chdir(os.path.join(path)) 
#%%Valida si existen las carpetas
try:
    os.makedirs("plots")
    os.makedirs("models")
except FileExistsError:
    pass

#%%gridsearchcv
#checar stratify
#sklearn.metrics.SCORERS.keys()
def gridsearchcv(X_train, X_test, y_train, y_test):
    ############
    # Scale numeric values
    num_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])
    
    preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('num', num_transformer, make_column_selector(pattern='EDAD'))
            ])
    ############
    
    pipe_steps = [
        #('scaler', StandardScaler()),
        ('preprocessor', preprocessor),
        ('SupVM', SVC(kernel='rbf',probability=True))
        ]
    
    param_grid= {
        'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 
        'SupVM__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50, 75, 100]
        }
    
    pipeline = Pipeline(pipe_steps)
    grid = GridSearchCV(pipeline, param_grid,refit = True,cv = 5, verbose = 3, n_jobs=-1,scoring='f1') #'balanced_accuracy'
    grid.fit(X_train, y_train)
    print("Best-Fit Parameters From Training Data:\n",grid.best_params_)
    grid_predictions = grid.best_estimator_.predict(X_test) 
    report = classification_report(y_test, grid_predictions, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(confusion_matrix(y_test, grid_predictions))
    return grid, report

#%%CASO 1: prediccion de hospitalizacion por covid
hosp_data = pd.read_csv("prediction_data/df_caso1.zip")
#Porcentaje de informacion del dataset
hosp_data = hosp_data.sample(frac=0.001)

#separar datos
X = hosp_data.loc[:, hosp_data.columns != 'TIPO_PACIENTE']
y = hosp_data.loc[:,'TIPO_PACIENTE']
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y, shuffle=True)
#%%Entrenamiento (opcional)
#---->train
grid, grid_report= gridsearchcv(X_train, X_test, y_train, y_test)
#guarda el modelo y su reporte
joblib.dump(grid, 'models/hosp_data_grid.pkl', compress = 1)
#joblib.dump(hosp_data_grid_report, 'models/hosp_data_grid_report.pkl', compress = 1)
grid_report.to_csv("models/hosp_data_grid_report.csv", index=True)
'''
Best-Fit Parameters From Training Data:
 {'SupVM__C': 500, 'SupVM__gamma': 1}
              precision    recall  f1-score    support
0              0.845872  0.964435  0.901271  478.00000
1              0.645833  0.269565  0.380368  115.00000
accuracy       0.829680  0.829680  0.829680    0.82968
macro avg      0.745852  0.617000  0.640819  593.00000
weighted avg   0.807078  0.829680  0.800253  593.00000
[[461  17]
 [ 84  31]]
'''
#%%importa el modelo y su rendimiento
grid = joblib.load('models/hosp_data_grid.pkl')
grid_report = pd.read_csv("models/hosp_data_grid_report.csv", index_col=0)
print("best score from grid search: %f" % grid.best_estimator_.score(X_test, y_test))
#%%Pipeline without gridsearchcv

num_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('num', num_transformer, ['EDAD'])])

svc_model = Pipeline([
    #('scaler', StandardScaler()),
    ('preprocessor', preprocessor),
    ('SupVM', SVC(kernel='rbf',probability=True))])

svc_model.set_params(**grid.best_params_)
svc_model.get_params("model")
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#%%se preprocesan los datos
X_train_scaled = pd.DataFrame(svc_model.named_steps['preprocessor'].fit_transform(X_train),columns = X_train.columns)
X_test_scaled = pd.DataFrame(svc_model.named_steps['preprocessor'].fit_transform(X_test),columns = X_test.columns)
#cambia el orden de edad y sexo
X_train_scaled[['EDAD','SEXO']]=X_train_scaled[['SEXO','EDAD']]
X_test_scaled[['EDAD','SEXO']]=X_test_scaled[['SEXO','EDAD']]

#%%
#prueba el modelo con X_test
y_test.iloc[20] 
X_test.iloc[20]

svc_model.named_steps['SupVM'].predict(X_test_scaled.iloc[20, :].values.reshape(1, -1))
svc_model.named_steps['SupVM'].predict_proba(X_test_scaled.iloc[20, :].values.reshape(1, -1))
#%%
import shap
# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(model = svc_model.named_steps['SupVM'].predict_proba, data = X_train_scaled, link = 'logit')
shap_values = explainer.shap_values(X = X_test_scaled, nsamples = 30, l1_reg="num_features(12)")

print(f'length of SHAP values: {len(shap_values)}')
print(f'Shape of each element: {shap_values[0].shape}')

#prediction and probability of model
X_test.iloc[0, :]
print(f'Prediction for 1st sample in X_test: ', svc_model.named_steps['SupVM'].predict(X_test_scaled.iloc[[0], :])[0])
print(f'Prediction probability for 1st sample in X_test: ', svc_model.named_steps['SupVM'].predict_proba(X_test_scaled.iloc[[0], :])[0])

# plot the SHAP values for the false (0) output of the first instance
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test_scaled.iloc[0,:], link="logit")

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test_scaled.iloc[0,:], link="logit")

#Explaining the Prediction for all samples in Test set
#no hospitalizado
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], X_test_scaled)

#si hospitalizado
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test_scaled)

#SHAP Summary Plots
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values[1], X_test_scaled)
shap.summary_plot(shap_values[0], X_test_scaled)

#SHAP Dependence Plots
shap.dependence_plot("HIPERTENSION", shap_values[1], X_test_scaled)
# %%
