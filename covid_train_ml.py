'''
robust versions of logistic regression
support vector machines
random forests
gradient boosted decision trees
'''
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
sns.set(color_codes=True)
df = pd.read_csv("covid_data.csv")
# %%
class pca():
    def __init__(self,  df=None, titulo="Unspecified", label_solo=None, label_y=None):
        self.df = df
        self.label_solo = str(label_solo)
        self.label_y = str(label_y)
        self.titulo = str(titulo)
        
        print(f"Numero de elementos de {label_solo}\n", df[label_solo].value_counts())
        df = df[df[label_solo] == 1]
        df.drop([label_solo], axis=1, inplace = True)
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
                    c=df[self.label_y],
                    cmap='rainbow',
                    marker='o',
                    s=2,
                    linewidths=0)
        #genera legend del target
        labels = np.unique(df[self.label_y])
        handles = [plt.Line2D([],[],marker=".", ls="", 
                              color=scatter.cmap(scatter.norm(yi))) for yi in labels]
        plt.legend(handles, labels)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.title(self.titulo)
        #plt.show()
        plt.savefig(self.titulo + "_2D.png",format='png', dpi=1200)
        y = df[self.label_y]
        return x_pca, y
    def pca_3D(self):
        sns.set_style("white")  
        df[self.label_y] = pd.Categorical(df[self.label_y])
        my_color = df[self.label_y].cat.codes
        df_PCA = df.drop([self.label_y], axis=1)
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
        labels = np.unique(df[self.label_y])
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
        y = df[self.label_y]
        return result, y
#%%los fallecidos de todos los contagiagos
data = pca(df, titulo="Grafica PCA defuncion por covid", label_solo="RESULTADO", label_y="BOOL_DEF")
X, y = data.pca_2D()
data.pca_3D()
#%%los intubados de todos los hospitalizados

#SOLO CASOS POSITIVOS COVID
df = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
df.drop(['RESULTADO'], axis=1, inplace = True)

data = pca(df, titulo="Grafica PCA de los intubados en hospitalizacion", label_solo="TIPO_PACIENTE", label_y="INTUBADO")
X, y = data.pca_2D()
data.pca_3D()

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# Create a SVC classifier
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
#SOLO CASOS POSITIVOS COVID
df = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
df.drop(['RESULTADO'], axis=1, inplace = True)
#lista de columnas
list(df)
#solo hospitalizados
df['TIPO_PACIENTE'].value_counts()
df = df[df.TIPO_PACIENTE == 1]
df.drop(['TIPO_PACIENTE'], axis=1, inplace = True)
list(df)
df['INTUBADO'].value_counts()
df['BOOL_DEF'].value_counts()