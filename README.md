# Análisis y Predicción de Riesgos por COVID-19 en México
![](https://img.shields.io/github/stars/CT-6282/COVID-19_Paper.svg) ![](https://img.shields.io/github/forks/CT-6282/COVID-19_Paper.svg) ![](https://img.shields.io/github/issues/CT-6282/COVID-19_Paper.svg)

Con este proyecto se pretende seguir la evolución y los patrones generados por el COVID-19 mediante gráficas y datos sencillos de visualizar para así generar conciencia de los riesgos que una persona dada pueda llegar a tener tomando en cuenta sus descriptores particulares. 

**Table of Contents**
- [Análisis y Predicción de Riesgos por COVID-19 en México](#Análisis-y-Predicción-de-Riesgos-por-COVID-19-en-México)
  * [Motivación](#Motivación)
  * [Dataset](#Dataset)
  * [Tecnologías usadas](#Tecnologías-usadas)
  * [Funciones](#Funciones)
  * [Instalación](#Instalación)
    + [Ejemplos de código](#Ejemplos-de-código)
	+ [Pruebas](#Pruebas)
	+ [¿Cómo usarlo?](#¿Cómo-usarlo?)
	+ [Discusión](#Discusión)
  * [Contribuye](#Contribuye)
  * [Créditos](#Créditos)
  * [Licencia](#Licencia)

## Motivación
Proporcionar a la gente con un análisis que evita mostrar conclusiones ambiguas acerca del estado actual del país para asistir una toma de decisiones objetiva tanto por parte de autoridades como de ciudadanos.

## Dataset
Se utilizaron los datos abiertos de la Dirección General de Epidemiología y la Secretaría de Salud

## Tecnologías usadas
- **Python** 3.6.11
- **Urlib:** Se utilizó para descargar el .zip
- **io:** Se utilizó para guardar el dataset en memoria
- **Zipfile:** Se utilizó para extraer el archivo .zip
- **Pandas:**  Se utilizó para procesamiento del dataset
- **Numpy:** Se utilizó para procesamiento del dataset
- **Seaborn:** Se utilizó para generar gráficas
- **Matplotlib:** Se utilizó para generar gráficas
- **Sklearn:** Se utilizó para hacer el entrenamiento y clasificación del dataset
- **Joblib:** Se utilizó para guardar e importar los modelos
- **Streamlit:** Se utilizó para generar una aplicación web dinámica

## Funciones
- covid_data.py se usa para descargar el dataset y hacer el preprocesamiento 
- covid_graficas.py genera las gráficas y las guarda adentro de la carpeta plots 
- ejemplo_train_ml.py ejecuta el clasificador con un dataset de prueba y genera graficas para visualizar  
- covid_train_ml.py realiza el entrenamiento y guarda los modelos adentro de la carpeta de models 

El proyecto utiliza los datos abiertos de la secretaría de salud, y se actualizan diariamente. Se puede utiliza abriendo la aplicación web.
El algoritmo de predicción se basa en casos específicos de un paciente contagiado con CoV-2 para poder tomar las medidas adecuadas y tratarse lo antes posible, como lo son:
- predecir en base a los descriptores si el paciente contagiado de CoV-2 necesitará hospitalización
- predecir en base a los descriptores la mortalidad del paciente contagiado de CoV-2 antes de estar internado en el hospital
- predecir en base a los descriptores la mortalidad del paciente contagiado de CoV-2 al estar internado en el hospital
- predecir en base a los descriptores la necesidad de la Unidad de Cuidados Intensivos del paciente contagiado de CoV-2 al estar internado en el hospital sin tener un diagnóstico de neumonía
- predecir en base a los descriptores la necesidad de la Unidad de Cuidados Intensivos del paciente contagiado de CoV-2 con un diagnóstico de neumonía  al estar internado en el hospital
- predecir en base a los descriptores la necesidad de un ventilador invasivo del paciente contagiado de CoV-2 sin un diagnóstico de neumonía y sin haber requerido de ICU al estar internado en el hospital
- predecir en base a los descriptores la necesidad de un ventilador invasivo del paciente contagiado de CoV-2 con un diagnóstico de neumonía y haber requerido de ICU al estar internado en el hospital

## Instalación
Para poder ejecutar los archivos de forma local se recomienda crear un virtual environment con python 3.6.11  por:  

- **Anaconda**

Instalación: https://docs.anaconda.com/anaconda/install/  

`conda create --name prueba python=3.6 `

`conda activate prueba `

`cd path/repostitorio `

`pip install -r requirements.txt `

- **pipenv**

Instalación: https://pipenv-fork.readthedocs.io/en/latest/basics.html 

`pip install pipenv`

`cd path/repositorio`

`pipenv --python 3.6`

`pipenv shell`

`pipenv install -r requirements.txt`

`sudo apt-get install python3-tk`

Para instalar las librarías necesarias se utiliza este comando `pip install –r /path/to/requirements.txt` dentro del environment activado.

### Pruebas
Para probar que se esta ejecutando correctamente de forma local corra los siguientes comandos.

`python covid_data.py` en la terminal debe de imprimir `Se ha generado el archivo .csv` si se ejecuto de forma correcta.

`python covid_graficas.py` en la terminal debe de imprimir el numero de casos de covid y debe de generar nuevas graficas adentro de la carpeta “plots” si se ejecuto de forma correcta.

`python ejemplo_train_ml.py` en la terminal para porbar si la librería sklearn esta instalada en la version adecuada y debe de generar unas graficas de resultado.

### ¿Cómo usarlo?
Para utlizar este proyecto hay 3 modalidades. 

La primera es abriendo la aplicación web http://52.205.82.130:8501/

La segunda es abriendo los archivos jupyter por binder https://notebooks.gesis.org/binder/jupyter/user/ct-6282-covid-19_paper-7hxj79qb/tree/notebooks

La ultima es descargar este repositorio y correrlo de forma local. Se puede correr en un ide (visual studio code, spyder, etc) los archivos .ipynb o .py. Para poder ejecutarlos se tiene que cambiar el path o la dirección de la carpeta donde se encuentra el repositorio para que pueda leer el dataset y generar las graficas . Para hacer el entrenamiento se recomienda usar un porcentaje de los datos  df = df.sample(frac=0.01) para que el tiempo de entrenamiento no sea muy tardado


## Contribuye
Si hay algun tipo de grafica útil nos lo puedes hacer saber para desarrollarla y subirla al repositorio, asi como algun algorimo de clasificación también lo podemos incluir.

## Créditos
- Hrisko, J. (2020, April 11). Visualizing COVID-19 Data in Python. Retrieved July 30, 2020, from https://makersportal.com/blog/2020/4/5/visualizing-covid-19-data-in-python
- Sarkar, T. (2020, March 31). Analyze NY Times Covid-19 Dataset. Retrieved July 30, 2020, from https://towardsdatascience.com/analyze-ny-times-covid-19-dataset-86c802164210
- Wollenstein-Betech, S., Cassandras, C. G., & Paschalidis, I. C. (2020). Personalized Predictive Models for Symptomatic COVID-19 Patients Using Basic Preconditions: Hospitalizations, Mortality, and the Need for an ICU or Ventilator. doi:10.1101/2020.05.03.20089813
- Malik, U. (n.d.). Implementing SVM and Kernel SVM with Python's Scikit-Learn. Retrieved July 30, 2020, from https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
- Bhattacharyya, S. (2019, September 26). Principal Component Analysis and SVM in a Pipeline with Python. Retrieved August 04, 2020, from https://towardsdatascience.com/visualizing-support-vector-machine-decision-boundary-69e7591dacea
- VanderPlas, J. (n.d.). In-Depth: Support Vector Machines. Retrieved July 30, 2020, from https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
- COVID-19 Tablero México. (n.d.). Retrieved July 30, 2020, from https://coronavirus.gob.mx/datos/
- Información referente a casos COVID-19 en México - Bases de datos COVID-19 - datos.gob.mx/busca. (n.d.). Retrieved July 30, 2020, from https://www.gob.mx/salud/documentos/datos-abiertos-152127

## Licencia
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
