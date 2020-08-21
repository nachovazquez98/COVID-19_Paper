#%%
'''
robust versions of logistic regression
support vector machines
random forests
gradient boosted decision trees
'''

import os, sys
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV 
import joblib
sns.set(color_codes=True)
from sklearn.pipeline import Pipeline

#%%abrir csv
path = "/home/nacho/Documents/coronavirus/COVID-19_Paper/"
#path = "D:\ricar\Documents\Development\Python\COVID-19_Paper"
os.chdir(os.path.join(path)) 
df = pd.read_csv("covid_data.csv.zip")
#df = pd.read_csv(r"D:\ricar\Documents\Development\Python\COVID-19_Paper\covid_data.csv.zip", encoding='utf-8') #path directo
#%%10% de los datos aleatorios
df = df.sample(frac=0.001)
#%%Valida si existen las carpetas
try:
    os.makedirs("plots")
    os.makedirs("models")
except FileExistsError:
    pass
#%%solamente
def solamente(df, columna, bool=None):
    if bool == None:
        bool = 1
    df = df[df[columna] == bool] #filtrar
    df.drop([columna], axis=1, inplace = True)
    return df


#%%gridsearchcv
#checar stratify
#sklearn.metrics.SCORERS.keys()
def gridsearchcv(X, y, n_pca=None):
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,
                                            test_size=0.2, 
                                            stratify=y, 
                                            #random_state=False,
                                            shuffle=True)
    pipe_steps = [('scaler', StandardScaler()), ('SupVM', SVC(kernel='rbf',probability=True))]
    param_grid= {
            'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 
            'SupVM__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
    }
    pipeline = Pipeline(pipe_steps)
    grid = GridSearchCV(pipeline, param_grid,refit = True,verbose = 3, n_jobs=-1,scoring='accuracy')
    grid.fit(X_train, Y_train)
    print("Best-Fit Parameters From Training Data:\n",grid.best_params_)
    grid_predictions = grid.predict(X_test) 
    report = classification_report(Y_test, grid_predictions, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(confusion_matrix(Y_test, grid_predictions))
    return grid, report, X_test, Y_test


#%%prediccion de hospitalizacion por covid - PCA
hosp_data = df.copy()
hosp_data = solamente(hosp_data,'RESULTADO')
hosp_data = hosp_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','TIPO_PACIENTE']]
hosp_data = hosp_data.reset_index(drop=True)
#separar datos
X = hosp_data.loc[:, hosp_data.columns != 'TIPO_PACIENTE']
y = hosp_data.loc[:,'TIPO_PACIENTE']
print(y.value_counts())
#---->train
hosp_data_grid, hosp_data_grid_report, X_test, Y_test = gridsearchcv(X,y)
#guarda el modelo y su reporte
#joblib.dump(hosp_data_svm.best_estimator_, 'models/hosp_data_svm.pkl', compress = 1)
joblib.dump(hosp_data_grid, 'models/hosp_data_grid.pkl', compress = 1)
hosp_data_grid_report.to_csv("models/hosp_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
hosp_data_grid_load = joblib.load('models/hosp_data_grid.pkl')
hosp_data_grid_report = pd.read_csv("models/hosp_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
hosp_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1)) 
hosp_data_grid.predict_proba(X_test.iloc[20,:].values.reshape(1,-1))

#%%Mortalidad de los contagiagos ANTES de ir al hospital
def_data = df.copy()
def_data = solamente(def_data,'TIPO_PACIENTE', bool=0) #revisar si mejora el rendimiento
def_data = solamente(def_data,'RESULTADO')
def_data = def_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','BOOL_DEF']]
X = def_data.loc[:, def_data.columns != 'BOOL_DEF']
y = def_data.loc[:,'BOOL_DEF']
#---->train
def_data_grid, def_data_grid_report, X_test, Y_test = gridsearchcv(X,y)
#guarda el modelo y su reporte
joblib.dump(def_data_grid, 'models/def_data_grid.pkl', compress = 1)
def_data_grid_report.to_csv("models/def_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
def_data_grid_load = joblib.load('models/def_data_grid.pkl')
def_data_grid_report = pd.read_csv("models/def_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
def_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1))


#%%Mortalidad de los contagiagos DEESPUES de ir al hospital
def_hosp_data = df.copy()
def_hosp_data = solamente(def_hosp_data,'TIPO_PACIENTE')
def_hosp_data = solamente(def_hosp_data,'RESULTADO')
def_hosp_data = def_hosp_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','INTUBADO','UCI','BOOL_DEF']]
X = def_hosp_data.loc[:, def_hosp_data.columns != 'BOOL_DEF']
y = def_hosp_data.loc[:,'BOOL_DEF']
#---->train
def_hosp_data_grid, def_hosp_data_grid_report, X_test, Y_test = gridsearchcv(X,y)
#guarda el modelo y su reporte
joblib.dump(def_hosp_data_grid, 'models/def_hosp_data_grid.pkl', compress = 1)
def_hosp_data_grid_report.to_csv("models/def_hosp_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
def_hosp_data_grid_load = joblib.load('models/def_hosp_data_grid.pkl')
def_hosp_data_grid_report = pd.read_csv("models/def_hosp_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
def_hosp_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1))

#%%Necesidad de ICU ANTES de saber si o no tiene neumonia
icu_data = df.copy()
icu_data = solamente(icu_data,'RESULTADO')
icu_data = icu_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','UCI']]
X = icu_data.loc[:, icu_data.columns != 'UCI']
y = icu_data.loc[:,'UCI']
#---->train
icu_data_grid, icu_data_grid_report, X_test, Y_test = gridsearchcv(X,y)
#guarda el modelo y su reporte
joblib.dump(icu_data_grid, 'models/icu_data_grid.pkl', compress = 1)
icu_data_grid_report.to_csv("models/icu_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
icu_data_grid_load = joblib.load('models/icu_data_grid.pkl')
icu_data_grid_report = pd.read_csv("models/icu_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
icu_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1))

#%%Necesidad de ICU despues de saber si o no tiene neumonia
icu_neum_data = df.copy()
icu_neum_data = solamente(icu_neum_data,'RESULTADO')
icu_neum_data = icu_neum_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','NEUMONIA','UCI']]
X = icu_neum_data.loc[:, icu_neum_data.columns != 'UCI']
y = icu_neum_data.loc[:,'UCI']
#---->train
icu_neum_data_grid, icu_neum_data_grid_report, X_test, Y_test = gridsearchcv(X,y)
#guarda el modelo y su reporte
joblib.dump(icu_neum_data_grid, 'models/icu_neum_data_grid.pkl', compress = 1)
icu_neum_data_grid_report.to_csv("models/icu_neum_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
icu_neum_data_grid_load = joblib.load('models/icu_neum_data_grid.pkl')
icu_neum_data_grid_report = pd.read_csv("models/icu_neum_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
def_hosp_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1))


#%%necesidad de ventilador antes de saber si desarrollo neumonia o necesita ICU
vent_data = df.copy()
vent_data = solamente(vent_data,'RESULTADO')
vent_data = vent_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','INTUBADO']]
X = vent_data.loc[:, vent_data.columns != 'INTUBADO']
y = vent_data.loc[:,'INTUBADO']
#---->train
vent_data_grid, vent_data_grid_report, X_test, Y_test = gridsearchcv(X,y)
#guarda el modelo y su reporte
joblib.dump(vent_data_grid, 'models/vent_data_grid.pkl', compress = 1)
vent_data_grid_report.to_csv("models/vent_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
vent_data_grid_load = joblib.load('models/vent_data_grid.pkl')
vent_data_grid_report = pd.read_csv("models/vent_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
vent_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1))

#%%necesidad de ventilador despues de saber si desarrollo neumonia o necesita ICU
vent_ucineum_data = df.copy()
vent_ucineum_data = solamente(vent_ucineum_data,'RESULTADO')
vent_ucineum_data = vent_ucineum_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','ICU','NEUMONIA','INTUBADO']]
X = vent_ucineum_data.loc[:, vent_ucineum_data.columns != 'INTUBADO']
y = vent_ucineum_data.loc[:,'INTUBADO']
#---->train
vent_ucineum_data_grid, vent_ucineum_data_grid_report, X_test, Y_test = gridsearchcv(X,y)
#guarda el modelo y su reporte
joblib.dump(vent_ucineum_data_grid, 'models/vent_ucineum_data_grid.pkl', compress = 1)
vent_ucineum_data_grid_report.to_csv("models/vent_ucineum_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
vent_ucineum_data_grid_load = joblib.load('models/vent_ucineumdata_grid.pkl')
vent_ucineum_data_grid_report = pd.read_csv("models/vent_ucineum_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
vent_ucineum_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1))