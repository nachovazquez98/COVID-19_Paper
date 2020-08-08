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
df = df.sample(frac=0.01)
#%%Valida si existen las carpetas
try:
    os.makedirs("plots")
    os.makedirs("models")
except FileExistsError:
    pass
# %%pca
class pca():
    def __init__(self,  df=None, titulo="Unspecified", label_y=None):
        self.df = df
        self.label_y = str(label_y)
        self.titulo = str(titulo)
        print(list(df))
        print(f"Numero de elementos de {label_y}\n", df[label_y].value_counts())
    def pca_2D(self):
        df_PCA = self.df.drop([self.label_y], axis=1)
        #instanciamos el metodo pca con 2 componentes
        pca = PCA(n_components=2)
        #encontramos los componentes principales usando 
        #el método de ajuste con 2 componentes
        #transformamos los datos scaled_data en 2 componentes con pca
        pca.fit(df_PCA)
        x_pca = pca.transform(df_PCA)
        ######
        #instanciamos un objeto para hacer PCA
        scaler = StandardScaler()
        #escalar los datos, estandarizarlos, para que cada
        #caracteristica tenga una varianza unitaria 
        scaler.fit(df_PCA)
        #aplicamos la reducción de rotación y dimensionalidad
        scaled_data = scaler.transform(df_PCA)
        pca = PCA().fit(scaled_data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.title('How many components are needed to describe the data.')
        ######
        print("Dimension de los features orginales: ", df_PCA.shape)
        print("Dimension de los features con 2 componentes", x_pca.shape)
        
        #visualizar los datos en 2 dimensiones
        #plt.figure(figsize=(8,6))
        fig, ax = plt.subplots()
        scatter = plt.scatter(x_pca[:,0],
                    x_pca[:,1],
                    c=self.df[self.label_y],
                    cmap='rainbow',
                    marker='o',
                    s=2,
                    linewidths=0)
        #genera legend del target
        labels = np.unique(self.df[self.label_y])
        handles = [plt.Line2D([],[],marker=".", ls="", 
                              color=scatter.cmap(scatter.norm(yi))) for yi in labels]
        plt.legend(handles, labels)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.title(self.titulo)
        #plt.show()
        plt.savefig("plots/"+self.titulo + "_2D.png",format='png', dpi=1200)
        y = self.df[self.label_y]
        return x_pca, y
    def pca_3D(self):
        sns.set_style("white")  
        self.df[self.label_y] = pd.Categorical(self.df[self.label_y])
        my_color = self.df[self.label_y].cat.codes
        df_PCA = self.df.drop([self.label_y], axis=1)
        pca = PCA(n_components=3)
        pca.fit(df_PCA)
        result=pd.DataFrame(pca.transform(df_PCA), 
                            columns=['PCA%i' % i for i in range(3)], 
                            index=df_PCA.index)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scat = ax.scatter(result['PCA0'], 
                   result['PCA1'], 
                   result['PCA2'], 
                   c=my_color, 
                   cmap='rainbow', 
                   s=2, marker="o",
                   linewidths=0)
        
        #genera legend del target
        labels = np.unique(self.df[self.label_y])
        handles = [plt.Line2D([],[],marker=".",ls="",
                                 color=scat.cmap(scat.norm(yi))) for yi in labels]               
        ax.legend(handles, labels)
        
        # make simple, bare axis lines through space:
        xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
         
        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(self.titulo)
        #plt.show()
        fig.tight_layout()
        plt.savefig("plots/"+self.titulo + "_3D.png", format='png', dpi=1200)
        y = self.df[self.label_y]
        return result, y


#%%solamente
def solamente(df, columna, bool=None):
    if bool == None:
        bool = 1
    df = df[df[columna] == bool] #filtrar
    df.drop([columna], axis=1, inplace = True)
    return df


#%%gridsearchcv
#checar stratify
def gridsearchcv(X, y, n_pca=None):
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,
                                            test_size=0.2, 
                                            stratify=y, 
                                            #random_state=False,
                                            shuffle=True)
    pipe_steps_pca = [('scaler', StandardScaler()),('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]
    param_grid_pca= {
        'pca__n_components': [n_pca], 
        'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 
        'SupVM__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
    }
    pipe_steps = [('scaler', StandardScaler()), ('SupVM', SVC(kernel='rbf'))]
    param_grid= {
            'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 
            'SupVM__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
    }
    if n_pca != None:
        pipeline = Pipeline(pipe_steps_pca)
        grid = GridSearchCV(pipeline, param_grid_pca,refit = True,verbose = 3, n_jobs=-1,probability=True)
    else:
        pipeline = Pipeline(pipe_steps)
        grid = GridSearchCV(pipeline, param_grid,refit = True,verbose = 3, n_jobs=-1,probability=True)
    grid.fit(X_train, Y_train)
    print ("Best-Fit Parameters From Training Data:\n",grid.best_params_)
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
#visualizacion pca
#hosp_pca = pca(hosp_data, titulo="Grafica PCA Hospitalizacion por COVID", label_y="TIPO_PACIENTE")
#hosp_pca.pca_2D(); hosp_pca.pca_3D()
#separar datos
X = hosp_data.loc[:, hosp_data.columns != 'TIPO_PACIENTE']
y = hosp_data.loc[:,'TIPO_PACIENTE']
#---->train
#hosp_data_grid, hosp_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=2)
hosp_data_grid, hosp_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None)
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

#%%Mortalidad de los contagiagos ANTES de ir al hospital
def_data = df.copy()
def_data = solamente(def_data,'TIPO_PACIENTE', bool=0) #revisar si mejora el rendimiento
def_data = solamente(def_data,'RESULTADO')
def_data = def_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','BOOL_DEF']]
X = def_data.loc[:, def_data.columns != 'BOOL_DEF']
y = def_data.loc[:,'BOOL_DEF']
#PCA
#def_pca = pca(df = def_data, titulo="Grafica PCA defuncion antes de hosp por covid", label_y="BOOL_DEF")
#def_pca.pca_2D(); def_pca.pca_3D()
#X, y = def_pca.pca_3D()
#---->train
def_data_grid, def_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None)
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
def_hosp_data_grid, def_hosp_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None)
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
icu_data_grid, icu_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None)
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
icu_neum_data_grid, icu_neum_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None)
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
vent_data_grid, vent_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None)
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
vent_ucineum_data_grid, vent_ucineum_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None)
#guarda el modelo y su reporte
joblib.dump(vent_ucineum_data_grid, 'models/vent_ucineum_data_grid.pkl', compress = 1)
vent_ucineum_data_grid_report.to_csv("models/vent_ucineum_data_grid_report.csv", index=True)
#importa el modelo y su rendimiento
vent_ucineum_data_grid_load = joblib.load('models/vent_ucineumdata_grid.pkl')
vent_ucineum_data_grid_report = pd.read_csv("models/vent_ucineum_data_grid_report.csv", index_col=0)
#prueba el modelo load con input sin preprocesamiento
Y_test.iloc[20]
vent_ucineum_data_grid_load.predict(X_test.iloc[20,:].values.reshape(1,-1))