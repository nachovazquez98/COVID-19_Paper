'''
robust versions of logistic regression
support vector machines
random forests
gradient boosted decision trees
'''
import os
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
sns.set(color_codes=True)
os.getcwd()
os.chdir('/home/nacho/Documents/coronavirus/COVID-19_Paper/')
df = pd.read_csv("covid_data.csv")
# %%
class pca():
    def __init__(self,  df=None, titulo="Unspecified", label_y=None):
        self.df = df
        self.label_y = str(label_y)
        self.titulo = str(titulo)
        print(list(df))
        print(f"Numero de elementos de {label_y}\n", df[label_y].value_counts())
    def pca_2D(self):
        df_PCA = self.df.drop([self.label_y], axis=1)
        #instanciamos un objeto para hacer PCA
        scaler = StandardScaler()
        #escalar los datos, estandarizarlos, para que cada
        #caracteristica tenga una varianza unitaria 
        scaler.fit(df_PCA)
        #aplicamos la reducción de rotación y dimensionalidad
        scaled_data = scaler.transform(df_PCA)
        #instanciamos el metodo pca con 2 componentes
        pca = PCA(n_components=2)
        #encontramos los componentes principales usando 
        #el método de ajuste con 2 componentes
        #transformamos los datos scaled_data en 2 componentes con pca
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)
        ######
        pca = PCA().fit(scaled_data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.title('How many components are needed to describe the data.')
        ######
        print("Dimension de los features orginales: ", scaled_data.shape)
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
        plt.savefig(self.titulo + "_2D.png",format='png', dpi=1200)
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
        plt.savefig(self.titulo + "_3D.png", format='png', dpi=1200)
        y = self.df[self.label_y]
        return result, y
#%%hyper_svm
def hyper_svm(X, y):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'kernel': ['rbf']} 
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    grid_predictions = grid.predict(X_test) 
    # print classification report 
    print(classification_report(y_test, grid_predictions)) 
    print(confusion_matrix(y_test, grid_predictions))
    return grid
#%%prediccion de hospitalizacion por covid - PCA
hosp_data = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
hosp_data.drop(['RESULTADO'], axis=1, inplace = True)
hosp_data_pca = hosp_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','TIPO_PACIENTE']]
hosp_pca = pca(hosp_data_pca, titulo="Grafica PCA Hospitalizacion por COVID", label_y="TIPO_PACIENTE")
X, y = hosp_pca.pca_2D()
hosp_pca.pca_3D()
#X, y = hosp_pca.pca_3D()
#---->train
hyper_svm(X,y)
#%%Mortalidad de los contagiagos ANTES de ir al hospital
def_data = df[df.TIPO_PACIENTE == 0]
def_data.drop(['TIPO_PACIENTE'], axis=1, inplace = True)
def_data = def_data[def_data.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
def_data.drop(['RESULTADO'], axis=1, inplace = True)
def_data = def_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','BOOL_DEF']]
def_pca = pca(df = def_data, titulo="Grafica PCA defuncion antes de hosp por covid", label_y="BOOL_DEF")
X, y = def_pca.pca_2D()
def_pca.pca_3D()
#X, y = def_pca.pca_3D()
#---->train
hyper_svm(X,y)
#%%Mortalidad de los contagiagos DEESPUES de ir al hospital
def_hosp_data = df[df.TIPO_PACIENTE == 1]
def_hosp_data.drop(['TIPO_PACIENTE'], axis=1, inplace = True)
def_hosp_data = def_hosp_data[def_hosp_data.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
def_hosp_data.drop(['RESULTADO'], axis=1, inplace = True)
def_hosp_data = def_hosp_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','BOOL_DEF','INTUBADO','UCI']]
def_hosp_pca = pca(df = def_data, titulo="Grafica PCA defuncion despues de hosp por covid", label_y="BOOL_DEF")
X, y = def_hosp_pca.pca_2D()
def_hosp_pca.pca_3D()
#X, y = def_pca.pca_3D()
#---->train
hyper_svm(X,y)
#%%
df.UCI.value_counts() #-->validar
#%%Necesidad de ICU ANTES de saber si o no tiene neumonia
icu_data = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
icu_data.drop(['RESULTADO'], axis=1, inplace = True)
icu_data = icu_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','UCI']]
icu_data_pca = pca(df = icu_data, titulo="Grafica PCA Intubado por covid antes de saber neumonia", label_y="UCI")
X, y = icu_data_pca.pca_2D()
icu_data_pca.pca_3D()
#X, y = icu_data_pca.pca_3D()
#---->train
hyper_svm(X,y)
#%%Necesidad de ICU despues de saber si o no tiene neumonia
icu_neum_data = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
icu_neum_data.drop(['RESULTADO'], axis=1, inplace = True)
icu_neum_data = icu_neum_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','NEUMONIA','UCI']]
icu_neum_pca = pca(df = icu_neum_data, titulo="Grafica PCA Intubado por covid despues de saber neumonia", label_y="UCI")
X, y = icu_neum_pca.pca_2D()
icu_neum_pca.pca_3D()
#X, y = icu_data_pca.pca_3D()
#---->train
hyper_svm(X,y)
#%%Ejemplo
# Here we are using inbuilt dataset of scikit learn 
from sklearn.datasets import load_breast_cancer 
  
# instantiating 
cancer = load_breast_cancer() 
  
# creating dataframe 
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names']) 
df['target'] = cancer['target']
cancer_pca = pca(df, titulo="cancer", label_y='target')
#X, y = cancer_pca.pca_2D()
X, y, pca = cancer_pca.pca_3D()
#---->train
grid = hyper_svm(X,y)
grid.predict([[-81.75387141, -65.15700805,  -2.87823954]])
grid.predict([[-273.82155725,   28.27681179,   -7.82953981]])
#grid_predictions_prueba = grid.predict(X_test.iloc[0,:].values.reshape(1,-1)) 
