#%%
'''
robust versions of logistic regression
support vector machines
random forests
gradient boosted decision trees

1) solo entrenar 10% de df y hacer pruebas
2) buscar otros algoritmos (agregar random forest classifier)
2.5) agregar simple PCA
3) agregar ui para acceder a los modelos entrenados
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
from sklearn.externals import joblib
sns.set(color_codes=True)
#%%abrir csv
os.getcwd()
os.chdir('/home/nacho/Documents/coronavirus/COVID-19_Paper/')
df = pd.read_csv("covid_data.csv")
#%%10% de los datos aleatorios
df = df.sample(frac=0.01)
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
    #%%hyper_svm
def hyper_svm(X, y):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'kernel': ['rbf']} 
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True) 
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    grid_predictions = grid.predict(X_test) 
    # print classification report 
    report = classification_report(y_test, grid_predictions, output_dict=True)
    print(report) 
    print(confusion_matrix(y_test, grid_predictions))
    return grid, report
#%%
def solamente(df, columna, bool=None):
    if bool == None:
        bool = 1
    df = df[df[columna] == bool] #filtrar
    df.drop([columna], axis=1, inplace = True)
    return df
#%%
def report_df(report, report_name):
    report = pd.DataFrame(report).transpose()
    print(report)
    report.to_csv("models/"+str(report_name)+".csv", index=True)
    return report   
#%%prediccion de hospitalizacion por covid - PCA
hosp_data = df.copy()
hosp_data = solamente(hosp_data,'RESULTADO')
hosp_data = hosp_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','TIPO_PACIENTE']]
hosp_pca = pca(hosp_data, titulo="Grafica PCA Hospitalizacion por COVID", label_y="TIPO_PACIENTE")
X, y = hosp_pca.pca_2D()
hosp_pca.pca_3D()
#X, y = hosp_pca.pca_3D()
#---->train
hosp_data_svm, hosp_data_svm_report = hyper_svm(X,y)
#graficar modelo 2d
plot_svm(hosp_data_svm)
#guarda el modelo y su reporte
joblib.dump(hosp_data_svm.best_estimator_, 'models/hosp_data_svm.pkl', compress = 1)
hosp_data_svm_report = report_df(hosp_data_svm_report, "hosp_data_svm_report")
#%%Mortalidad de los contagiagos ANTES de ir al hospital
def_data = df.copy()
def_data = solamente(def_data,'TIPO_PACIENTE', bool=0)
def_data = solamente(def_data,'RESULTADO')
def_data = def_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','BOOL_DEF']]
def_pca = pca(df = def_data, titulo="Grafica PCA defuncion antes de hosp por covid", label_y="BOOL_DEF")
X, y = def_pca.pca_2D()
def_pca.pca_3D()
#X, y = def_pca.pca_3D()
#---->train
def_data_svm, def_data_svm_report = hyper_svm(X,y)
#guarda el modelo y su reporte
joblib.dump(def_data_svm.best_estimator_, 'def_data_svm.pkl', compress = 1)
def_data_svm_report = report_df(def_data_svm_report, "def_data_svm_report")
#%%Mortalidad de los contagiagos DEESPUES de ir al hospital
def_hosp_data = df.copy()
def_hosp_data = solamente(def_hosp_data,'TIPO_PACIENTE')
def_hosp_data = solamente(def_hosp_data,'RESULTADO')
def_hosp_data = def_hosp_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','BOOL_DEF','INTUBADO','UCI']]
def_hosp_pca = pca(df = def_hosp_data, titulo="Grafica PCA defuncion despues de hosp por covid", label_y="BOOL_DEF")
X, y = def_hosp_pca.pca_2D()
def_hosp_pca.pca_3D()
#X, y = def_pca.pca_3D()
#---->train
def_hosp_data_svm, def_hosp_data_svm_report = hyper_svm(X,y)
#guarda el modelo y su reporte
joblib.dump(def_hosp_data_svm.best_estimator_, 'def_hosp_data_svm.pkl', compress = 1)
def_hosp_data_svm_report = report_df(def_hosp_data_svm_report, "def_hosp_data_svm_report")
#%%
df.UCI.value_counts() #-->validar
#%%Necesidad de ICU ANTES de saber si o no tiene neumonia
icu_data = df.copy()
icu_data = solamente(icu_data,'RESULTADO')
icu_data = icu_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','UCI']]
icu_data_pca = pca(df = icu_data, titulo="Grafica PCA Intubado por covid antes de saber neumonia", label_y="UCI")
X, y = icu_data_pca.pca_2D()
icu_data_pca.pca_3D()
#X, y = icu_data_pca.pca_3D()
#---->train
icu_data_svm, icu_data_svm_report = hyper_svm(X,y)
#guarda el modelo y su reporte
joblib.dump(def_hosp_data_svm.best_estimator_, 'icu_data_svm.pkl', compress = 1)
icu_data_svm_report = report_df(icu_data_svm_report, "icu_data_svm_report")
#%%Necesidad de ICU despues de saber si o no tiene neumonia
icu_neum_data = df.copy()
icu_neum_data = solamente(icu_neum_data,'RESULTADO')
icu_neum_data = icu_neum_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','NEUMONIA','UCI']]
icu_neum_pca = pca(df = icu_neum_data, titulo="Grafica PCA Intubado por covid despues de saber neumonia", label_y="UCI")
X, y = icu_neum_pca.pca_2D()
icu_neum_pca.pca_3D()
#X, y = icu_data_pca.pca_3D()
#---->train
icu_neum_data_svm, icu_neum_data_svm_report = hyper_svm(X,y)
#guarda el modelo y su reporte
joblib.dump(icu_neum_data_svm.best_estimator_, 'icu_neum_data_svm.pkl', compress = 1)
icu_neum_data_svm_report = report_df(icu_neum_data_svm_report, "icu_neum_data_svm_report")
#%%necesidad de ventilador antes de saber si desarrollo neumonia o necesita ICU
vent_data = df.copy()
vent_data = solamente(vent_data,'RESULTADO')
vent_data = vent_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','INTUBADO']]
vent_data_pca = pca(df = vent_data, titulo="Grafica PCA Intubado por covid antes de saber neumonia/UCI", label_y="INTUBADO")
X, y = vent_data_pca.pca_2D()
vent_data_pca.pca_3D()
#X, y = icu_data_pca.pca_3D()
#---->train
vent_data_svm, vent_data_svm_report = hyper_svm(X,y)
#guarda el modelo y su reporte
joblib.dump(vent_data_svm.best_estimator_, 'vent_data_svm.pkl', compress = 1)
vent_data_svm_report = report_df(vent_data_svm_report, "vent_data_svm_report")
#%%necesidad de ventilador despues de saber si desarrollo neumonia o necesita ICU
vent_ucineum_data = df.copy()
vent_ucineum_data = solamente(vent_ucineum_data,'RESULTADO')
vent_ucineum_data = vent_ucineum_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','INTUBADO','ICU','NEUMONIA']]
vent_ucineum_data_pca = pca(df = vent_ucineum_data, titulo="Grafica PCA Intubado por covid despues de saber neumonia/UCI", label_y="INTUBADO")
X, y = vent_ucineum_data_pca.pca_2D()
vent_ucineum_data_pca.pca_3D()
#X, y = icu_data_pca.pca_3D()
#---->train
vent_ucineum_data_svm, vent_ucineum_data_svm_report = hyper_svm(X,y)
#guarda el modelo y su reporte
joblib.dump(vent_ucineum_data_svm.best_estimator_, 'vent_ucineum_data_svm.pkl', compress = 1)
vent_ucineum_data_svm_report = report_df(vent_ucineum_data_svm_report, "vent_ucineum_data_svm_report")
#%%Ejemplo
# Here we are using inbuilt dataset of scikit learn 
from sklearn.datasets import load_breast_cancer 
# instantiating 
cancer = load_breast_cancer() 
# creating dataframe 
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names']) 
df['target'] = cancer['target']
df.head()
cancer_pca = pca(df, titulo="cancer", label_y='target')
#X, y = cancer_pca.pca_2D()
X, y= cancer_pca.pca_3D()
#---->train
cancer_svm, cancer_svm_report = hyper_svm(X,y)
print(cancer_svm.best_params_)
############################
#####grafica modelo en 2d########
plot_svm(cancer_svm)
###########################
#############################3
#guarda el mejor modelo y su rendimiento
joblib.dump(cancer_svm.best_estimator_, 'models/cancer_svm.pkl', compress = 1)
cancer_svm_report = report_df(cancer_svm_report, "cancer_svm_report")
#importa el modelo y su rendimiento
cancer_svm = joblib.load('models/cancer_svm.pkl')
cancer_svm_report = pd.read_csv("models/cancer_svm_report.csv")
#probar input del modelo
input = df.iloc[222,:-1].values.reshape(1,-1) #selecciona la fila 222 y todas las columnas excepto label
input_label = df.iloc[222,-1]
#pendiente transformar input a 3 dimensiones para pedict
#print(pca.components_) 
cancer_svm.predict([[-81.75387141, -65.15700805,  -2.87823954]])
cancer_svm.predict([[-273.82155725,   28.27681179,   -7.82953981]])
#grid_predictions_prueba = cancer_grid.predict(X_test.iloc[0,:].values.reshape(1,-1)) 
#%%Pipeline Steps are StandardScaler, PCA and SVM 
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_train, X_test, Y_train, Y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    test_size=0.25, 
                                                    stratify=cancer.target, 
                                                    random_state=30)

pipe_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]

check_params= {
    'pca__n_components': [2], 
    'SupVM__C': [0.1, 0.5, 1, 10,30, 40, 50, 75, 100, 500, 1000], 
    'SupVM__gamma' : [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}

pipeline = Pipeline(pipe_steps)

create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
create_grid.fit(X_train, Y_train)
print ("score for %d fold CV := %3.2f" %(cv, create_grid.score(X_test, Y_test)))
print ("!!!!!!!! Best-Fit Parameters From Training Data !!!!!!!!!!!!!!")
print (create_grid.best_params_)
print ("grid best params: ", create_grid.best_params_)
# %% plot smv 2d   
def plot_svm(model):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=2, cmap='rainbow',marker='o',linewidths=0)
    plot_svc_decision_function(model)
    plt.scatter(model.best_estimator_.support_vectors_[:, 0],
                model.best_estimator_.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none');
    
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.best_estimator_.support_vectors_[:, 0],
                   model.best_estimator_.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


