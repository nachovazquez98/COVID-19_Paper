#%%
'''
http://epistasislab.github.io/tpot/api/#classification
https://machinelearningmastery.com/tpot-for-automated-machine-learning-in-python/
'''
import os, sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pathlib
from tpot import TPOTClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from notify_run import Notify
#%%abrir csv
#path = "/home/nacho/Documents/coronavirus/COVID-19_Paper/"
path = "/lustre/home/idvperez/COVID-19_Paper/"
os.chdir(os.path.join(path))

#path = pathlib.Path(__file__).parent.absolute()
#os.chdir(path)
print(os.getcwd())

#data_percentage = 0.01
data_percentage = 1
#%%Valida si existen las carpetas
try:
    os.makedirs("tpot/models")
except FileExistsError:
    pass
#%%
notify = Notify()
channel = notify.register()
endpoint = channel.endpoint
print(endpoint) # https://notify.run/<channel_code>
channel_page = channel.channel_page
print(channel_page) # https://notify.run/c/<channel_page_code>
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
i = 1
for subdir, dirs, files in os.walk(str_path+'prediction_data'):
    notify.send('Empezo el proceso TPOT') 
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
            X, y = X.values, y.values
            X, y = X.astype('float32'), y.astype('float32')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify=y, shuffle=True)
            #---->train
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
            tpot = TPOTClassifier(generations=5, population_size=50, scoring='balanced_accuracy', verbosity = 3, n_jobs = -1, cv= cv)
            tpot.fit(X_train, y_train)
            predictions = tpot.predict(X_test) 
            report = classification_report(y_test, predictions, output_dict=True)
            report = pd.DataFrame(report).transpose()
            #guarda el modelo y su reporte
            #joblib.dump(tpot, 'tpot/'+file_name+'_tpot_model.pkl', compress = 1)
            report.to_csv('tpot/models/'+file_name+'_tpot_report.csv', index=True)
            tpot.export('tpot/models/'+file_name+'_tpot_pipeline.py')
            notify.send("Termino dataset # " + str(i))
            i = 1 + i

notify.send('Finalizo el proceso TPOT') 
