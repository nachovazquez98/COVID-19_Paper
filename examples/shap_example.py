#%%
'''
https://www.kaggle.com/kritidoneria/explainable-ai-eli5-lime-and-shap
https://medium.com/analytics-vidhya/shap-part-2-kernel-shap-3c11e7a971b1

'''
import os, sys
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn import metrics
import joblib
from sklearn.pipeline import Pipeline
from gridsearchcv import Gridsearchcv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import sklearn_evaluation
#%%importa el modelo y su rendimiento
grid = joblib.load('models/hosp_data_grid.pkl')
grid_report = pd.read_csv("models/hosp_data_grid_report.csv", index_col=0)
#print("best score from grid search: %f" % grid.best_estimator_.score(X_test, y_test))
#print("model prediction: ", grid.predict(pd.DataFrame(X_train.iloc[20, :].values.reshape(1, -1), columns = X_train.columns)))
#print("model prediction probability: ",grid.predict_proba(pd.DataFrame(X_train.iloc[20, :].values.reshape(1, -1), columns = X_train.columns)))
#%%Pipeline without gridsearchcv
num_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('num', num_transformer, ['EDAD'])])

model = Pipeline([
    ('preprocessor', preprocessor),
    #('SupVM', SVC(kernel='rbf',probability=True))
    #('rf', RandomForestClassifier())
    #('SupVM', LogisticRegression(solver='liblinear'))
    ('estimator__clf__available_models__gb', GradientBoostingClassifier())
    ])

#no son iguales
#grid.get_params().keys() ->quitar basura en el string
#model.get_params().keys()

model.set_params(**grid.best_params_)
#model.set_params(**{'SupVM__fit_intercept': True, 'SupVM__max_iter': 100, 'SupVM__penalty': 'l1', 'SupVM__tol': 0.0001})
#model.get_params(**{'fit_intercept': False, 'max_iter': 10, 'penalty': 'l1', 'tol': 1e-05})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
#%%se preprocesan los datos
X_train_scaled = pd.DataFrame(model.named_steps['preprocessor'].fit_transform(X_train),columns = X_train.columns)
X_test_scaled = pd.DataFrame(model.named_steps['preprocessor'].fit_transform(X_test),columns = X_test.columns)
#cambia el orden de edad y sexo
X_train_scaled[['EDAD','SEXO']]=X_train_scaled[['SEXO','EDAD']]
X_test_scaled[['EDAD','SEXO']]=X_test_scaled[['SEXO','EDAD']]

#%%
#prueba el modelo con X_test
y_test.iloc[20] 
X_test.iloc[20]

model.named_steps['SupVM'].predict(X_test_scaled.iloc[20, :].values.reshape(1, -1))
model.named_steps['SupVM'].predict_proba(X_test_scaled.iloc[20, :].values.reshape(1, -1))
#%%
import shap
# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(model = model.named_steps['SupVM'].predict_proba, data = X_train_scaled, link = 'logit')

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

print(f'Prediction for 1st sample in X_test: ', model.named_steps['SupVM'].predict(X_test_scaled.iloc[[0], :])[0])
print(f'Prediction probability for 1st sample in X_test: ', model.named_steps['SupVM'].predict_proba(X_test_scaled.iloc[[0], :])[0])

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
