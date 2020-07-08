#https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from socket import timeout
import pandas as pd
import numpy as np
import os

#Se extrae el csv de la web
url = "http://187.191.75.115/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip"
try:
    resp = urlopen(url, timeout=10).read()
    zipfile = ZipFile(BytesIO(resp))
    extracted_file = zipfile.open(zipfile.namelist()[0])
    df = pd.read_csv(extracted_file, encoding = "ISO-8859-1")
    print("Se abrio el link")
except (ConnectionResetError, timeout) as error:
    print(error.args)
    #print("ConnectionResetError or Timeout")
    print("Se abrira el archivo zip...")

    zipfile = ZipFile('/home/nacho/Documents/coronavirus/rangel/datos_abiertos_covid19.zip', 'r')
    extracted_file = zipfile.open(zipfile.namelist()[0])
    df = pd.read_csv(extracted_file, encoding = "ISO-8859-1")

finally:    
    #df = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
    #df.drop(['RESULTADO','FECHA_ACTUALIZACION', 'ID_REGISTRO', 'ORIGEN', 'SECTOR', 'ENTIDAD_UM', 'MIGRANTE', 'PAIS_ORIGEN', 'PAIS_NACIONALIDAD'], axis=1, inplace = True)
    
    #Se eliminan las columnas innecesarias
    df.drop(['FECHA_ACTUALIZACION', 'ID_REGISTRO', 'ORIGEN', 'SECTOR', 'MIGRANTE', 'PAIS_ORIGEN', 'PAIS_NACIONALIDAD'], axis=1, inplace = True)
    #%%
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
    ################################################
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
    ####################################################
    #columna defuncion cambia a 0 a los no fallecidos
    df['BOOL_DEF'] = df['BOOL_DEF'].replace(['9999-99-99'], 0)
    #columna defuncion cambia a 1 los fallecidos
    df.loc[df["BOOL_DEF"] != 0, ["BOOL_DEF"]] = 1
    #%%
    #eliminar columna FECHA_INGRESO, FECHA_SINTOMAS y FECHA_DEF
    #df.drop(['FECHA_INGRESO', 'FECHA_SINTOMAS', 'FECHA_DEF'], axis=1, inplace = True)
        
    #Se cambian los valores de 97, 98 y 99 a 2. Se escribe 3 veces por columna a modificar debido a unos errores que encontramos, modificaban datos equivocados
    df.loc[df['INTUBADO'] == 97, ['INTUBADO']] = 2;df.loc[df['INTUBADO'] == 98, ['INTUBADO']] = 2;df.loc[df['INTUBADO'] == 99, ['INTUBADO']] = 2
    df.loc[df['NEUMONIA'] == 97, ['NEUMONIA']] = 2;df.loc[df['NEUMONIA'] == 98, ['NEUMONIA']] = 2;df.loc[df['NEUMONIA'] == 99, ['NEUMONIA']] = 2
    df.loc[df['EMBARAZO'] == 97, ['EMBARAZO']] = 2;df.loc[df['EMBARAZO'] == 98, ['EMBARAZO']] = 2;df.loc[df['EMBARAZO'] == 99, ['EMBARAZO']] = 2
    df.loc[df['HABLA_LENGUA_INDIG'] == 97, ['HABLA_LENGUA_INDIG']] = 2;df.loc[df['HABLA_LENGUA_INDIG'] == 98, ['HABLA_LENGUA_INDIG']] = 2;df.loc[df['HABLA_LENGUA_INDIG'] == 99, ['HABLA_LENGUA_INDIG']] = 2
    df.loc[df['DIABETES'] == 97, ['DIABETES']] = 2;df.loc[df['DIABETES'] == 98, ['DIABETES']] = 2;df.loc[df['DIABETES'] == 99, ['DIABETES']] = 2
    df.loc[df['EPOC'] == 97, ['EPOC']] = 2;df.loc[df['EPOC'] == 98, ['EPOC']] = 2;df.loc[df['EPOC'] == 99, ['EPOC']] = 2
    df.loc[df['ASMA'] == 97, ['ASMA']] = 2;df.loc[df['ASMA'] == 98, ['ASMA']] = 2;df.loc[df['ASMA'] == 99, ['ASMA']] = 2
    df.loc[df['INMUSUPR'] == 97, ['INMUSUPR']] = 2;df.loc[df['INMUSUPR'] == 98, ['INMUSUPR']] = 2;df.loc[df['INMUSUPR'] == 99, ['INMUSUPR']] = 2
    df.loc[df['HIPERTENSION'] == 97, ['HIPERTENSION']] = 2;df.loc[df['HIPERTENSION'] == 98, ['HIPERTENSION']] = 2;df.loc[df['HIPERTENSION'] == 99, ['HIPERTENSION']] = 2
    df.loc[df['OTRA_COM'] == 97, ['OTRA_COM']] = 2;df.loc[df['OTRA_COM'] == 98, ['OTRA_COM']] = 2;df.loc[df['OTRA_COM'] == 99, ['OTRA_COM']] = 2
    df.loc[df['CARDIOVASCULAR'] == 97, ['CARDIOVASCULAR']] = 2;df.loc[df['CARDIOVASCULAR'] == 98, ['CARDIOVASCULAR']] = 2;df.loc[df['CARDIOVASCULAR'] == 99, ['CARDIOVASCULAR']] = 2
    df.loc[df['OBESIDAD'] == 97, ['OBESIDAD']] = 2;df.loc[df['OBESIDAD'] == 98, ['OBESIDAD']] = 2;df.loc[df['OBESIDAD'] == 99, ['OBESIDAD']] = 2
    df.loc[df['RENAL_CRONICA'] == 97, ['RENAL_CRONICA']] = 2;df.loc[df['RENAL_CRONICA'] == 98, ['RENAL_CRONICA']] = 2;df.loc[df['RENAL_CRONICA'] == 99, ['RENAL_CRONICA']] = 2
    df.loc[df['TABAQUISMO'] == 97, ['TABAQUISMO']] = 2;df.loc[df['TABAQUISMO'] == 98, ['TABAQUISMO']] = 2;df.loc[df['TABAQUISMO'] == 99, ['TABAQUISMO']] = 2
    df.loc[df['OTRO_CASO'] == 97, ['OTRO_CASO']] = 2;df.loc[df['OTRO_CASO'] == 98, ['OTRO_CASO']] = 2;df.loc[df['OTRO_CASO'] == 99, ['OTRO_CASO']] = 3
    df.loc[df['UCI'] == 97, ['UCI']] = 2;df.loc[df['UCI'] == 98, ['UCI']] = 2
    
    #Se cambian los valores de 1, 2 e incluso 3 a 0 y 1. Se separa para mantener más claro el proceso
    #En SEXO son: 0 - Hombre, 1 - Mujer, 2 - No especificado
    #En OTRO_CASO se cambiaron los datos: 1 - 1, 2|97|98 - 0, 99 - 2
    #En nacionalidad los datos son: 1- 1, 2|99 - 0 
    df.loc[df['SEXO'] == 2, ['SEXO']] = 0
    df.loc[df['SEXO'] == 3, ['SEXO']] = 2
    df.loc[df['INTUBADO'] == 2, ['INTUBADO']] = 0
    df.loc[df['NEUMONIA'] == 2, ['NEUMONIA']] = 0
    df.loc[df['EMBARAZO'] == 2, ['EMBARAZO']] = 0
    df.loc[df['HABLA_LENGUA_INDIG'] == 2, ['HABLA_LENGUA_INDIG']] = 0
    df.loc[df['DIABETES'] == 2, ['DIABETES']] = 0
    df.loc[df['EPOC'] == 2, ['EPOC']] = 0
    df.loc[df['ASMA'] == 2, ['ASMA']] = 0
    df.loc[df['INMUSUPR'] == 2, ['INMUSUPR']] = 0
    df.loc[df['HIPERTENSION'] == 2, ['HIPERTENSION']] = 0
    df.loc[df['OTRA_COM'] == 2, ['OTRA_COM']] = 0
    df.loc[df['CARDIOVASCULAR'] == 2, ['CARDIOVASCULAR']] = 0
    df.loc[df['OBESIDAD'] == 2, ['OBESIDAD']] = 0
    df.loc[df['RENAL_CRONICA'] == 2, ['RENAL_CRONICA']] = 0
    df.loc[df['TABAQUISMO'] == 2, ['TABAQUISMO']] = 0
    df.loc[df['OTRO_CASO'] == 2, ['OTRO_CASO']] = 0
    df.loc[df['OTRO_CASO'] == 99, ['OTRO_CASO']] = 2    
    df.loc[df['UCI'] == 2, ['UCI']] = 0
    df.loc[df['TIPO_PACIENTE'] == 1, ['TIPO_PACIENTE']] = 0
    df.loc[df['TIPO_PACIENTE'] == 2, ['TIPO_PACIENTE']] = 1
    df.loc[df['TIPO_PACIENTE'] == 99, ['TIPO_PACIENTE']] = 0
    df.loc[df['NACIONALIDAD'] == 2, ['NACIONALIDAD']] = 0
    df.loc[df['NACIONALIDAD'] == 99, ['NACIONALIDAD']] = 0
    
    #eliminar los hombres embarazados
    df.drop(df[(df['SEXO'] ==0) & (df['EMBARAZO'] ==1)].index, inplace = True)
    #verificacion
    df['SEXO'][(df['SEXO'] ==0) & (df['EMBARAZO'] ==1)]
    
    #Se imprime el dataframe
    #print(df)
    #%%
    df.to_csv('covid_data.csv', index=False)
    print("Se ha generado el archivo .csv")


