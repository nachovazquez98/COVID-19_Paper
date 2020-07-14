#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:37:12 2020

@author: nacho
"""
import sys, os
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV 
import joblib
from sklearn.pipeline import Pipeline
#%%           
pathname = os.path.dirname(sys.argv[0]) 
fullpath = os.path.abspath(pathname)       
print('path del codigo: ', fullpath) 
os.chdir(fullpath) #cambia directorio de trabajo en la dir del codigo
os.getcwd()
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
# %% plot smv 2d   
def plot_svm_2d(grid, X_test, Y_test):
    scaler1 = StandardScaler()
    scaler1.fit(X_test)
    X_test_scaled = scaler1.transform(X_test)
    
    
    pca1 = PCA(n_components=2)
    X_test_scaled_reduced = pca1.fit_transform(X_test_scaled)
    
    
    svm_model = SVC(kernel='rbf', C=float(grid.best_params_['SupVM__C']), 
                    gamma=float(grid.best_params_['SupVM__gamma']))
    
    classify = svm_model.fit(X_test_scaled_reduced, Y_test)
    
    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    
    def make_meshgrid(x, y, h=.1):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))#,
                             #np.arange(z_min, z_max, h))
        return xx, yy
    
    X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    fig, ax = plt.subplots(figsize=(12,9))
    fig.patch.set_facecolor('white')
    cdict1={0:'lime',1:'deeppink'}
    
    Y_tar_list = Y_test.tolist()
    yl1= [int(target1) for target1 in Y_tar_list]
    labels1=yl1
     
    labl1={0:'Malignant',1:'Benign'}
    marker1={0:'*',1:'d'}
    alpha1={0:.8, 1:0.5}
    
    for l1 in np.unique(labels1):
        ix1=np.where(labels1==l1)
        ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])
    
    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none', 
               edgecolors='navy', label='Support Vectors')
    
    plot_contours(ax, classify, xx, yy,cmap='seismic', alpha=0.4)
    plt.legend(fontsize=15)
    
    plt.xlabel("1st Principal Component",fontsize=14)
    plt.ylabel("2nd Principal Component",fontsize=14)
    
    #plt.savefig('ClassifyMalignant_Benign2D_Decs_FunctG10.png', dpi=300)
    plt.show()
#%%
def gridsearchcv(X, y, n_pca):
    X_train, X_test, Y_train, Y_test = train_test_split(X,y,
                                                        test_size=0.2, 
                                                        #stratify=y, 
                                                        #random_state=False,
                                                        shuffle=True)
    pipe_steps = [('scaler', StandardScaler()),('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]
    param_grid= {
        'pca__n_components': [n_pca], 
        'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 
        'SupVM__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
    }
    pipeline = Pipeline(pipe_steps)
    for cv in range(4,6):
        grid = GridSearchCV(pipeline, param_grid,refit = True,verbose = 3, cv=cv)
        grid.fit(X_train, Y_train)
        print ("score for %d fold CV := %3.2f" %(cv, grid.score(X_test, Y_test)))
    print ("Best-Fit Parameters From Training Data:\n",grid.best_params_)
    grid_predictions = grid.predict(X_test) 
    report = classification_report(Y_test, grid_predictions, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(confusion_matrix(Y_test, grid_predictions))
    return grid, report, X_test, Y_test
#%%Ejemplo
# Here we are using inbuilt dataset of scikit learn 
from sklearn.datasets import load_breast_cancer 
# instantiating 
cancer = load_breast_cancer() 
# creating dataframe 
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names']) 
df['target'] = cancer['target']
df.head()
#visualizacion pca
cancer_pca = pca(df, titulo="cancer", label_y='target')
cancer_pca.pca_2D(); cancer_pca.pca_3D()
#separar datos
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
#train
cancer_grid, cancer_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=2)
############################
#####grafica modelo en 2d###
plot_svm_2d(cancer_grid, X_test, Y_test)
############################
############################
#guarda el mejor modelo y su rendimiento
joblib.dump(cancer_grid.best_estimator_, 'models/cancer_grid.pkl', compress = 1)
#joblib.dump(cancer_grid, 'models/cancer_grid.pkl', compress = 1)
cancer_grid_report.to_csv("models/cancer_grid_report.csv", index=True)
#importa el modelo y su rendimiento
cancer_grid_load = joblib.load('models/cancer_grid.pkl')
cancer_grid_report = pd.read_csv("models/cancer_svm_report.csv", index_col=0)
#probar input del modelo
input = df.iloc[222,:-1].values.reshape(1,-1) #selecciona la fila 222 y todas las columnas excepto label
input_label = df.iloc[222,-1]
cancer_grid_load.predict(input)
#grid_predictions_prueba = cancer_grid.predict(X_test.iloc[0,:].values.reshape(1,-1)) 
