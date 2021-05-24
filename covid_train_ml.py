#%%
'''
robust versions of logistic regression
support vector machines
random forests
gradient boosted decision trees
https://www.kdnuggets.com/2020/06/simplifying-mixed-feature-type-preprocessing-scikit-learn-pipelines.html
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html
https://kiwidamien.github.io/introducing-the-column-transformer.html
'''
import os, sys
import pandas as pd
import numpy as np
import joblib
from gridsearchcv import Gridsearchcv
from sklearn.model_selection import train_test_split
import pathlib
#%%abrir csv
path = "/home/nacho/Documents/coronavirus/COVID-19_Paper/"
os.chdir(os.path.join(path)) 

#path = pathlib.Path(__file__).parent.absolute()
#os.chdir(path)

data_percentage = 0.01
#data_percentage = 1
#%%Valida si existen las carpetas
try:
    os.makedirs("plots")
    os.makedirs("models")
except FileExistsError:
    pass
#%%iter pred files
def pred_label(filename): 
    if filename.find('df_caso0') != -1:
        label = 'hosp_critica'
    if filename.find('caso1') != -1:
        label = 'TIPO_PACIENTE'
    if filename.find('caso2') != -1 or filename.find('caso3') != -1 or filename.find('df_caso_3_1') != -1 or filename.find('df_caso_3_2') != -1 or filename.find('df_caso_3_3') != -1:
        label = 'BOOL_DEF'
    if filename.find('caso5') != -1 or filename.find('df_caso5_1') != -1:
        label = 'UCI'
    if filename.find('caso6') != -1 or filename.find('caso7') != -1 or filename.find('df_caso_7_1') != -1 or filename.find('df_caso_7_2') != -1 or filename.find('df_caso_7_3') != -1:
        label = 'INTUBADO'
    return label

str_path = str(path)
print(str_path)
for subdir, dirs, files in os.walk(str_path+'prediction_data'):
    for file in files:
        if file.endswith(".zip"):
            file_path = subdir + "/" + file
            file_name = file.split('.', 1)[0]
            print(file_name)
            df_data = pd.read_csv(file_path)
            df_data = df_data.sample(frac=data_percentage)
            #separar datos
            label = pred_label(file_name)
            print(label)
            X = df_data.loc[:, df_data.columns != label]
            y = df_data.loc[:, label]
            print(y.value_counts())
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify=y, shuffle=True)
            #---->train
            grid, df_grid, grid_report= Gridsearchcv(X_train, X_test, y_train, y_test, randomizedsearch = True)
            #guarda el modelo y su reporte
            joblib.dump(grid, 'models/'+file_name+'_grid.pkl', compress = 1)
            grid_report.to_csv('models/'+file_name+'_grid_report.csv', index=True)
            df_grid.to_csv('models/'+file_name+'_df_grid.csv', index=True)
