#https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import urllib.request
from socket import timeout
import logging
import pandas as pd
import numpy as np
import os
import datetime
import requests

#Se extrae el csv de la web
#url = "http://187.191.75.115/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip"
url = 'http://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip'
path = '/home/nacho/Documents/coronavirus/rangel/datos_abiertos_covid19.zip' #del archivo .zip
#path = "/app"

def filter_only_positive(df):
    df = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por solo los que dieron positivo

def filter_exclude_columns(df):
    #df.drop(['RESULTADO','FECHA_ACTUALIZACION', 'ID_REGISTRO', 'ORIGEN', 'SECTOR', 'ENTIDAD_UM', 'MIGRANTE', 'PAIS_ORIGEN', 'PAIS_NACIONALIDAD'], axis=1, inplace = True)
    df.drop(['FECHA_ACTUALIZACION', 'ID_REGISTRO', 'ORIGEN', 'MIGRANTE', 'PAIS_ORIGEN', 'PAIS_NACIONALIDAD','MUNICIPIO_RES','ENTIDAD_NAC'], axis=1, inplace = True) #Se eliminan las columnas innecesarias

def date_preprocessing(df):
    #convierte a tipo fecha fecha_sintoma 
    df['FECHA_SINTOMAS'] = pd.to_datetime(df['FECHA_SINTOMAS'])
    #restar columna FECHA_INGRESO menos FECHA_SINTOMAS y guardar en columna dias_dif
    df['FECHA_INGRESO'] = pd.to_datetime(df['FECHA_INGRESO'])
    df['DIAS_DIF_HOSP'] = (df['FECHA_INGRESO'] - df['FECHA_SINTOMAS'])
    df.DIAS_DIF_HOSP = df.DIAS_DIF_HOSP.dt.days
    #eliminar todos los dias negativos
    df.drop(df[df['DIAS_DIF_HOSP'] < 0].index, inplace = True)
    #verificacion
    df['DIAS_DIF_HOSP'][df['DIAS_DIF_HOSP'] < 0]
    #df['DIAS_DIF_HOSP'].astype(int)

def filter_negative_dates(df):
    #hace una copia ed fecha_def a dias_dif_def
    df['BOOL_DEF'] = df["FECHA_DEF"].copy()
    #CREAR COLUMNA DE NUMERO DE DIAS DE SINTOMAS A FALLECIMIENTO
    #crea columna dias desde sintomas a fallecido
    #remplazar en fecha_def 9999 con nan
    df["FECHA_DEF"] = df['FECHA_DEF'].replace(['9999-99-99'], np.nan)
    #convertir fecha_def a tipo de dato fecha
    df['FECHA_DEF'] = pd.to_datetime(df['FECHA_DEF'])
    #restar defcha def menos fecha_sintomas
    df['DIAS_DIF_DEF'] = (df['FECHA_DEF'] - df['FECHA_SINTOMAS'])
    df.DIAS_DIF_DEF = df.DIAS_DIF_DEF.dt.days
    df['DIAS_DIF_DEF'] = df['DIAS_DIF_DEF'].replace([np.nan], 0)
    df['DIAS_DIF_DEF'] = df['DIAS_DIF_DEF'].astype(int)
    #validar si hay dias negativos y eliminarlos
    df.drop(df[df['DIAS_DIF_DEF'] < 0].index, inplace = True)
    #verificacion
    df['DIAS_DIF_DEF'][df['DIAS_DIF_DEF'] < 0]

def filter_binary_status(df):
    #columna defuncion cambia a 0 a los no fallecidos
    df['BOOL_DEF'] = df['BOOL_DEF'].replace(['9999-99-99'], 0)
    #columna defuncion cambia a 1 los fallecidos
    df.loc[df["BOOL_DEF"] != 0, ["BOOL_DEF"]] = 1

def filter_unused_dates(df):
    #eliminar columna FECHA_INGRESO, FECHA_SINTOMAS y FECHA_DEF
    df.drop(['FECHA_INGRESO', 'FECHA_SINTOMAS', 'FECHA_DEF'], axis=1, inplace = True)

def change_to_two(df):
    #Se cambian los valores de 97, 98 y 99 a 2. Se escribe 3 veces por columna a modificar debido a unos errores que encontramos, modificaban datos equivocados
    change_to_two_dictionary = ['EMBARAZO', 'RENAL_CRONICA', 'DIABETES', 'INMUSUPR', 'EPOC', 'OBESIDAD', 'OTRO_CASO', 'HIPERTENSION', 'TABAQUISMO', 'CARDIOVASCULAR', 'ASMA', 'OTRA_COM']
    for condition in change_to_two_dictionary: 
        df.loc[df[condition] == 97, [condition]] = 2;df.loc[df[condition] == 98, [condition]] = 2;df.loc[df[condition] == 99, [condition]] = 2

def remove_non_conclusive(df):
    non_conclusive_dictionary = ['SEXO'] #'TIPO_PACIENTE', 'INTUBADO', 'UCI', 'NEUMONIA',
    for condition in non_conclusive_dictionary:
        df.drop(df[df[condition] == 97].index, inplace = True)
        df.drop(df[df[condition] == 98].index, inplace = True)
        df.drop(df[df[condition] == 99].index, inplace = True)

def binary_values(df):
    #Se cambian los valores de 1, 2 e incluso 3 a 0 y 1. Se separa para mantener más claro el proceso
    #En SEXO son: 0 - Hombre, 1 - Mujer, 2 - No especificado
    #En OTRO_CASO se cambiaron los datos: 1 - 1, 2|97|98 - 0, 99 - 2
    #En nacionalidad los datos son: 1- 1, 2|99 - 0 
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

def filter_pregnant_men(df):
    #eliminar los hombres embarazados
    df.drop(df[(df['SEXO'] ==0) & (df['EMBARAZO'] ==1)].index, inplace = True)
    #verificacion
    df['SEXO'][(df['SEXO'] ==0) & (df['EMBARAZO'] ==1)]

def print_df(df):
#Se imprime el dataframe    
    #print(df)
    filename = 'covid_data'
    compression_options = dict(method='zip', archive_name=f'{filename}.csv')
    df.to_csv(f'{filename}.zip', compression=compression_options,index=False)
    print("Se ha generado el archivo .csv")

#Se ejecutan las funciones
try: #Se obtiene el archivo más reciente
    resp = urlopen(url, timeout=10).read() #Se omite el uso de una función en este segmento para evitar errores con las variables
    zipfile = ZipFile(BytesIO(resp))
    extracted_file = zipfile.open(zipfile.namelist()[0])
    df = pd.read_csv(extracted_file, encoding = "ISO-8859-1")
    print("Se está descargando la versión más actual del archivo...")
    _ = df
except (ConnectionResetError, timeout) as error: #Si se falla al obtener el archivo más reciente se usa el último registro local
    print(error.args) #Se omite el uso de una función en este segmento para evitar errores con las variables
    #print("ConnectionResetError or Timeout")
    print("Conexión fallida, se usará el último archivo local...")
    zipfile = ZipFile(path, 'r')
    extracted_file = zipfile.open(zipfile.namelist()[0])
    df = pd.read_csv(extracted_file, encoding = "ISO-8859-1")
    _ = df
finally:
    #filter_only_positive(df)
    filter_exclude_columns(df)
    date_preprocessing(df)
    filter_negative_dates(df)
    filter_binary_status(df)
    #filter_unused_dates(df)
    change_to_two(df)
    remove_non_conclusive(df)
    binary_values(df)
    filter_pregnant_men(df)
    print_df(df)
