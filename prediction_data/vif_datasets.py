from statsmodels.stats.outliers_influence import variance_inflation_factor    
import os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pathlib

path = "/home/nacho/Documents/coronavirus/COVID-19_Paper/"
#path = "/lustre/home/idvperez/COVID-19_Paper/"
os.chdir(os.path.join(path)) 

os.makedirs("prediction_data/reduced_dataset", exist_ok = True)

def calculate_vif_(X, thresh=100):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    #print(X.columns[variables])
    return X[cols[variables]]

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
            #separar datos
            label = pred_label(file_name)
            print(label)
            X = df_data.loc[:, df_data.columns != label]
            y = df_data.loc[:, label]
            print(y.value_counts())
            #appplt vif
            new_df = calculate_vif_(X, thresh=100)
            #save new df
            new_df[str(label)] = y
            #guarda el modelo
            print(new_df.columns)
            compression_options = dict(method='zip', archive_name=f'{file_name}.csv')
            new_df.to_csv(f'prediction_data/{file_name}.zip', compression=compression_options,index=False) 
