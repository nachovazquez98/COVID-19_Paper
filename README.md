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
- **Heroku:** Se utilizópara hostear la aplicación web en un contenedor de linux

## Funciones
El proyecto utliza los datos abiertos de la secretaría de salud, y se actualizan diariamente. Se puede utlizar abriendo la aplicación web.
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

Para instalar las librerias necesarias se utliza este comando `pip install –r /path/to/requirements.txt` dentro del environment activado.

### Ejemplos de código
Primero para conseguir el dataset, se descarga el .zip, despues se guarda en la memoria,  se extrae y se convierte a formato Dataframe.
```python
url = 'http://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip' 
resp = urlopen(url, timeout=10).read() #Se omite el uso de una función en este segmento para evitar errores con las variables 
zipfile = ZipFile(BytesIO(resp)) 
extracted_file = zipfile.open(zipfile.namelist()[0]) 
df = pd.read_csv(extracted_file, encoding = "ISO-8859-1") 
```
Para reallizar el preprocesamiento se eliminan colmnas, los valores se convierten a 0:No y 1:Si para poder manipular el archivo en las graficas y el entrenamiento, y otras operaciones como procesar las fechas y eliminar información invalida.

```python
df.drop(['FECHA_ACTUALIZACION', 'ID_REGISTRO', 'ORIGEN', 'SECTOR', 'MIGRANTE', 'PAIS_ORIGEN', 'PAIS_NACIONALIDAD'], axis=1, inplace = True) #Se eliminan las columnas innecesarias 

binary_values_dictionary = ['RESULTADO', 'SEXO', 'INTUBADO', 'NEUMONIA', 'EMBARAZO', 'HABLA_LENGUA_INDIG', 'DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'OTRO_CASO', 'UCI', 'NACIONALIDAD'] 

for condition in binary_values_dictionary: 
	df.loc[df[condition] == 2, [condition]] = 0 
	df.loc[df[condition] == 97, [condition]] = 0 
	df.loc[df[condition] == 98, [condition]] = 0 
	df.loc[df[condition] == 3, [condition]] = 2 
	df.loc[df[condition] == 99, [condition]] = 0 
	df.loc[df['TIPO_PACIENTE'] == 1, ['TIPO_PACIENTE']] = 0 
	df.loc[df['TIPO_PACIENTE'] == 2, ['TIPO_PACIENTE']] = 1 
	df.loc[df['TIPO_PACIENTE'] == 99, ['TIPO_PACIENTE']] = 0 
```

Al realizar las graficas se filtro solamente los positivos de COVID.

```python
df = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
df.drop(['RESULTADO'], axis=1, inplace = True)

#En esta grafica se filtra a todas las defunciones y se usa la columna edad para hacer una grafica de distribución
def grafica6():
	fig, ax = plt.subplots()
	plot_date(ax)
	df_solodef = df.loc[df.BOOL_DEF == 1]
	sns.distplot(df_solodef['EDAD']).set_title("Muertes de COVID-19 por edades en Mexico")
```

Para realizar el entrenamiento y clasificación se uso una pipeline y realiza todas las combinaciones de los hiperparametros y devuelve el mejor modelo asi como su redimiento.

```python
def gridsearchcv(X, y): 
	X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, stratify=y, shuffle=True) 
	pipe_steps = [('scaler', StandardScaler()), ('SupVM', SVC(kernel='rbf'))] 
	param_grid= { 
	'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],  
	'SupVM__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50] 
	} 
	pipeline = Pipeline(pipe_steps) 
	grid = GridSearchCV(pipeline, param_grid,refit = True,verbose = 3, n_jobs=-1,probability=True) 
	grid.fit(X_train, Y_train) 
	print("Best-Fit Parameters From Training Data:\n",grid.best_params_) 
	grid_predictions = grid.predict(X_test)  
	report = classification_report(Y_test, grid_predictions, output_dict=True) 
	report = pd.DataFrame(report).transpose() 
	print(report) 
	print(confusion_matrix(Y_test, grid_predictions)) 
return grid, report, X_test, Y_test 
```

Se manipula el dataset para poder genear los datos y sus etiquetas y enviarselas a la funcion gridsearchcv() para despues guardar e importar los modelos para poder utilizarlos en la interfaz web.

```python
hosp_data = df.copy() 
hosp_data = solamente(hosp_data,'RESULTADO') 
hosp_data = hosp_data.loc[:,['EDAD','EMBARAZO','RENAL_CRONICA','DIABETES','INMUSUPR','EPOC','OBESIDAD','OTRO_CASO','HIPERTENSION','TABAQUISMO','CARDIOVASCULAR','ASMA','SEXO','TIPO_PACIENTE']] 
hosp_data = hosp_data.reset_index(drop=True) 
#separar datos 
X = hosp_data.loc[:, hosp_data.columns != 'TIPO_PACIENTE'] 
y = hosp_data.loc[:,'TIPO_PACIENTE'] 
#---->train 
hosp_data_grid, hosp_data_grid_report, X_test, Y_test = gridsearchcv(X,y, n_pca=None) 
#guarda el modelo y su reporte 
joblib.dump(hosp_data_grid, 'models/hosp_data_grid.pkl', compress = 1) 
hosp_data_grid_report.to_csv("models/hosp_data_grid_report.csv", index=True) 
#importa el modelo y su rendimiento 
hosp_data_grid_load = joblib.load('models/hosp_data_grid.pkl') 
hosp_data_grid_report = pd.read_csv("models/hosp_data_grid_report.csv", index_col=0)
```

### Pruebas
Para probar que se esta ejecutnado correctamente de forma local corra los siguientes comandos.

`python covid_data.py` en la terminal debe de imprimir `Se ha generado el archivo .csv` si se ejecuto de forma correcta.

`python covid_graficas.py` en la terminal debe de imprimir el numero de casos de covid y debe de generar nuevas graficas adentro de la carpeta “plots” si se ejecuto de forma correcta.

`python ejemplo_train_ml.py` en la terminal para porbar si la librería sklearn esta instalada en la version adecuada y debe de generar unas graficas de resultado.

### ¿Cómo usarlo?
Para utlizar este proyecto hay 3 modalidades. 

La primera es abriendo la aplicación web https://covid19-analytics-app.herokuapp.com/

La siguiente es abrir los archivos .ipynb en este repositorio o en google collab para evitar la instalacion y configuración de las librerías. En el caso de google collab se tiene que subir el archivo del notebook en File/Open drive notebook y el dataset de forma manual en la ventana izquiera, en el boton de carpeta y adentro de /content la opcion “upload to session storage”

La ultima es descargar este repositorio y correrlo de forma local. Se puede correr en un ide (visual studio code, spyder, etc) los archivos .ipynb o .py. Para poder ejecutarlos se tiene que cambiar el path o la direccion de la carpeta donde se encuentra el repositorio para que pueda leer el dataset y generar las graficas . Para hacer el entrenamiento se recomienda usar un porcentaje de los datos  df = df.sample(frac=0.01) para que el tiempo de entrenamiento no sea muy tardado

### Discusión
Debido al alto costo computacional del entrenamiento con el dataset, no se pudieron realizar pruebas para mejorar la precision, ya sea mejorando los parametros del algoritmo de clasificacion o acomodar la informacion de la manera adecuada, como por ejemplo descartar a la gente que no ha tenido neumonia para predecir si va a necesitar un ventilador, etc. La forma del acomodo de los datos del entrenamiento se baso en este articulo https://www.medrxiv.org/content/10.1101/2020.05.03.20089813v1.full.pdf mas no entrenamos el modelo con el 100% de los datos debido su complejidad. 

Queremos realizar las pruebas al eliminar los datos invalidos del dataset como lo son NO APLICA, SE IGNORA, NO ESPECIFICADO en lugar de convertilos a falso y comparar su rendimiento. En un futuro tambien se pretende comparar diferentes algoritmos de clasificacion como lo son: robust versions of logistic regression, random forests, gradient boosted decision trees, y usar en la aplicacion web el modelo con el mejor rendimiento.

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
- Información referente a casos COVID-19 en México - Bases de datos COVID-19 - datos.gob.mx/busca. (n.d.). Retrieved July 30, 2020, from https://datos.gob.mx/busca/dataset/informacion-referente-a-casos-covid-19-en-mexico/resource/e8c7079c-dc2a-4b6e-8035-08042ed37165

## Licencia
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
