'''
falta grafica de acumulados y de mapa de correlacion
'''
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
sns.set(color_codes=True)
import os, datetime
from collections import Counter
from matplotlib.offsetbox import AnchoredText
os.getcwd()
os.chdir('/home/nacho/Documents/coronavirus/COVID-19_Paper/')
df = pd.read_csv("covid_data.csv")
#%%
#SOLO CASOS POSITIVOS COVID
df = df[df.RESULTADO == 1] #En caso de que se quiera filtrar por s{olo los que dieron positivo
df.drop(['RESULTADO'], axis=1, inplace = True)
#lista de columnas
list(df)
#%%
def plot_date(ax):
    txtbox = ax.text(0.0, 0.975, datetime.datetime.now().strftime('%b %d, %Y %H:%M'), transform=ax.transAxes, fontsize=7,
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='w',alpha=0.5)) 
    txtbox.set_x(1.0-(txtbox.figure.bbox.bounds[2]-(txtbox.clipbox.bounds[2]-txtbox.clipbox.bounds[0]))/txtbox.figure.bbox.bounds[2])
#PreAnalisis
def grafica1():
    print("Entidades de residencia con caso de covid:\n", df['ENTIDAD_RES'].value_counts())
    #
    fig, ax = plt.subplots() 
    plot_date(ax)
    ax.bar((df['ENTIDAD_RES'].value_counts()).index, (df['ENTIDAD_RES'].value_counts()).values) 
    ax.set_title('Entidades de residencia con caso de covid') 
    ax.set_xlabel('ENTIDAD_RES') 
    ax.set_ylabel('No. casos')
    fig.tight_layout()
    plt.savefig('entidades_casos_pos.png', format='png', dpi=1200)
grafica1()
#
def grafica2():
    print("Municipios de residencia con mas casos de covid:\n", df['MUNICIPIO_RES'].value_counts())
    #
    fig, ax = plt.subplots() 
    plot_date(ax)
    ax.bar((df['MUNICIPIO_RES'].value_counts())[:10].index, (df['MUNICIPIO_RES'].value_counts())[:10].values) 
    ax.set_title('10 Municipios de residencia con mas casos de covid') 
    ax.set_xlabel('MUNICIPIO_RES') 
    ax.set_ylabel('No. casos')
    ax.set_xticks((df['MUNICIPIO_RES'].value_counts())[:10].index)
    fig.tight_layout()
    plt.savefig('municipios_casos_pos.png', format='png', dpi=1200)
grafica2()
#
def grafica3():
    print("Pacientes con covid ambulatorios o hospitalizados:\n", df['TIPO_PACIENTE'].value_counts())
    #
    fig, ax = plt.subplots() 
    plot_date(ax)
    ax.bar((df['TIPO_PACIENTE'].value_counts()).index, (df['TIPO_PACIENTE'].value_counts()).values) 
    ax.set_title('Pacientes con covid ambulatorios o hospitalizados:') 
    ax.set_xlabel('Ambulatorios - Hospitalizados') 
    ax.set_ylabel('No. casos')
    ax.set_xticks((df['TIPO_PACIENTE'].value_counts()).index)
    fig.tight_layout()
    plt.savefig('amb_hosp_casos_pos.png', format='png', dpi=1200)
grafica3()
#
def grafica4():
    df['TIPO_PACIENTE'].value_counts()
    df_aux = df.loc[df.TIPO_PACIENTE == 1]
    df_aux.drop(['TIPO_PACIENTE'], axis=1, inplace = True)
    print("Pacientes Hospitalizados Intubados:\n", df_aux['INTUBADO'].value_counts())
    print("Porcentaje de Intubados Hospitalizados: ",((df['INTUBADO'].value_counts()).values[1]/(df['INTUBADO'].value_counts()).values[0])*100)
    #
    fig, ax = plt.subplots() 
    plot_date(ax)
    ax.bar((df_aux['INTUBADO'].value_counts()).index, (df_aux['INTUBADO'].value_counts()).values) 
    ax.set_title('Pacientes con covid hospitalizados intubados:') 
    ax.set_xlabel('No Intubados - Intubados') 
    ax.set_ylabel('No. casos')
    ax.set_xticks((df_aux['INTUBADO'].value_counts()).index)
    fig.tight_layout()
    plt.savefig('hosp_intubados_pos.png', format='png', dpi=1200)
grafica4()
#
def grafica5():
    print("Casos con covid fallecidos:\n", df['BOOL_DEF'].value_counts())
    print("Porcentaje de mortalidad: ",((df['BOOL_DEF'].value_counts()).values[1]/(df['BOOL_DEF'].value_counts()).values[0])*100)
    #
    fig, ax = plt.subplots() 
    plot_date(ax)
    ax.bar((df['BOOL_DEF'].value_counts()).index, (df['BOOL_DEF'].value_counts()).values) 
    ax.set_title('Pacientes con covid fallecidos:') 
    ax.set_xlabel('No fallecidos - fallecidos') 
    ax.set_ylabel('No. casos')
    ax.set_xticks((df['BOOL_DEF'].value_counts()).index)
    fig.tight_layout()
    plt.savefig('def_pos.png', format='png', dpi=1200)
grafica5()
#
#Solo los fallecidos
def grafica6():
    df_solodef = df.loc[df.BOOL_DEF == 1]
    sns.distplot(df_solodef['EDAD']).set_title("Muertes por COVID-19 en Mexico")  
    plt.savefig('def_edad_histograma.png', format='png', dpi=1200)
grafica6()
def grafica7():
    df_solodef = df.loc[df.BOOL_DEF == 1]
    sns.distplot(df_solodef['DIAS_DIF_DEF']).set_title("Dias entre los primeros sintomas y defuncion")      
grafica7()
def grafica8():
    sns.distplot(df['DIAS_DIF_HOSP']).set_title("Dias entre los primeros sintomas y hospitalizacion")      
grafica8()
def grafica9():
    df['edad_rango'] = pd.cut(x=df['EDAD'], bins=[0,17,44,64,74,max(df['EDAD'])], labels=['0-17','18-44','45-64','65-74','+75'])
    g = sns.catplot(x="edad_rango", y="BOOL_DEF", hue="SEXO", data=df,
                height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("Defuncion")
    plt.savefig('barplot_defuncion_edad.png', format='png', dpi=1200)
grafica9()
def grafica9():
    x= ['covid_rate', 'hosp_rate', 'death_rate']
    y = [len(df), len((df[df.TIPO_PACIENTE == 1])), len((df[df.BOOL_DEF == 1]))]
    g = sns.catplot(x=x, y=y, data=df, height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("No. casos")
    plt.title("Casos de COVID en Mexico")
    plt.xlabel(None)
    plt.savefig('barplot_casos_hos_def.png', format='png', dpi=1200)
grafica9()
def grafica10():
    df['edad_rango'] = pd.cut(x=df['EDAD'], bins=[0,17,44,64,74,max(df['EDAD'])], labels=['0-17','18-44','45-64','65-74','+75'])
    g = sns.catplot(x="edad_rango", y="TIPO_PACIENTE", hue="SEXO", data=df,
                height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("Hospitalizacion")
    g.set_titles("Hospitalizacion por rangos de edad")
    plt.savefig('barplot_hospitalizacion_edad.png', format='png', dpi=1200)
grafica10()
def grafica11():
    df['edad_rango'] = pd.cut(x=df['EDAD'], bins=[0,17,44,64,74,max(df['EDAD'])], labels=['0-17','18-44','45-64','65-74','+75'])
    labels = df['edad_rango'].cat.categories.tolist()
    covid_rate, hosp_rate, death_rate = [],[], []
    for i in range(len(labels)):
        covid_rate.append(len(df[df.edad_rango == df['edad_rango'].cat.categories.tolist()[i]]))
        hosp_rate.append(len(df[(df.edad_rango == df['edad_rango'].cat.categories.tolist()[i]) & (df.TIPO_PACIENTE == 1)]))
        death_rate.append(len(df[(df.edad_rango == df['edad_rango'].cat.categories.tolist()[i]) & (df.BOOL_DEF == 1)]))
    x = np.arange(len(labels)) #pocisiones de labels
    width = 0.30 #ancho de las barras
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/3, covid_rate, width, label="Casos COVID")
    bar2 = ax.bar(x + 2*(width/3), hosp_rate, width, label="Casos hosp")
    bar3 = ax.bar(x + 5*(width/3), death_rate, width, label="Casos muertes")
    ax.set_ylabel("No. de Casos")
    ax.set_title("Casos de COVID en Mexico por rangos de edad")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    autolabel(bar1); autolabel(bar2); autolabel(bar3)
    #plt.show()
    plot_date(ax)
    fig.tight_layout()
    plt.savefig("Casos de COVID en Mexico por rangos de edad.png", format='png', dpi=1200)
grafica11()
def grafica12():
    labels = ['Hombre', 'Mujer']
    covid_rate, hosp_rate, death_rate = [],[], []
    for i in range(len(labels)):
        covid_rate.append(len(df[df.SEXO == i]))
        hosp_rate.append(len(df[(df.SEXO == i) & (df.TIPO_PACIENTE == 1)]))
        death_rate.append(len(df[(df.SEXO == i) & (df.BOOL_DEF == 1)]))
    x = np.arange(len(labels)) #pocisiones de labels
    width = 0.30 #ancho de las barras
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/3, covid_rate, width, label="Casos COVID")
    bar2 = ax.bar(x + 2*(width/3), hosp_rate, width, label="Casos hosp")
    bar3 = ax.bar(x + 5*(width/3), death_rate, width, label="Casos muertes")
    ax.set_ylabel("No. de Casos")
    ax.set_title("Casos de COVID en Mexico por sexo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    autolabel(bar1); autolabel(bar2); autolabel(bar3)
    #plt.show()
    plot_date(ax)
    fig.tight_layout()
    plt.savefig("Casos de COVID en Mexico por sexo.png", format='png', dpi=1200)
grafica12()
def grafica13():
    x= ['EPOC_', 'UCI_', 'INTUBADO_']
    y = [len((df[df.EPOC == 1])), len((df[df.UCI == 1])),len((df[df.INTUBADO == 1]))]
    g = sns.catplot(x=x, y=y, data=df, height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("No. casos")
    plt.title("Casos de COVID en Mexico")
    plt.xlabel(None)
    plt.savefig('barplot_casos_hos_def.png', format='png', dpi=1200)
grafica13()
def grafica14():
    df['edad_rango'] = pd.cut(x=df['EDAD'], bins=[0,17,44,64,74,max(df['EDAD'])], labels=['0-17','18-44','45-64','65-74','+75'])
    labels = df['edad_rango'].cat.categories.tolist()
    EPOC_rate, UCI_rate, INTUBADO_rate = [],[], []
    for i in range(len(labels)):
        EPOC_rate.append(len(df[(df.edad_rango == df['edad_rango'].cat.categories.tolist()[i]) & (df.EPOC == 1)]))
        UCI_rate.append(len(df[(df.edad_rango == df['edad_rango'].cat.categories.tolist()[i]) & (df.UCI == 1)]))
        INTUBADO_rate.append(len(df[(df.edad_rango == df['edad_rango'].cat.categories.tolist()[i]) & (df.INTUBADO == 1)]))
    x = np.arange(len(labels)) #pocisiones de labels
    width = 0.30 #ancho de las barras
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/3, EPOC_rate, width, label="Casos EPOC")
    bar2 = ax.bar(x + 2*(width/3), UCI_rate, width, label="Casos UCI")
    bar3 = ax.bar(x + 5*(width/3), INTUBADO_rate, width, label="Casos Intubado")
    ax.set_ylabel("No. de Casos")
    ax.set_title("Casos de COVID en Mexico por rangos de edad")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    autolabel(bar1); autolabel(bar2); autolabel(bar3)
    #plt.show()
    plot_date(ax)
    fig.tight_layout()
    plt.savefig("Casos de COVID hospitalarios en Mexico por rangos de edad.png", format='png', dpi=1200)
grafica14()
def grafica15():
    labels = ['Hombre', 'Mujer']
    covid_rate, hosp_rate, death_rate = [],[], []
    for i in range(len(labels)):
        covid_rate.append(len(df[(df.SEXO == i) & (df.EPOC == 1)]))
        hosp_rate.append(len(df[(df.SEXO == i) & (df.UCI == 1)]))
        death_rate.append(len(df[(df.SEXO == i) & (df.INTUBADO == 1)]))
    x = np.arange(len(labels)) #pocisiones de labels
    width = 0.30 #ancho de las barras
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/3, covid_rate, width, label="Casos EPOC")
    bar2 = ax.bar(x + 2*(width/3), hosp_rate, width, label="Casos UCI")
    bar3 = ax.bar(x + 5*(width/3), death_rate, width, label="Casos Intubado")
    ax.set_ylabel("No. de Casos")
    ax.set_title("Casos de COVID en Mexico por sexo")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
    #Attach a text label above each bar in *rects*, displaying its height.
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    autolabel(bar1); autolabel(bar2); autolabel(bar3)
    #plt.show()
    plot_date(ax)
    fig.tight_layout()
    plt.savefig("Casos de COVID hospitalarios en Mexico por sexo.png", format='png', dpi=1200)
grafica15()

def casos_nuevos_indiv(titulo, columna_fecha, npol, estado):
    if estado != False:
        df_aux = df.copy()
        df_aux.drop(df_aux[(df_aux['ENTIDAD_UM'] != estado)].index, inplace = True)
    elif estado == False:
        df_aux = df.copy()
    fechas = Counter(df_aux[columna_fecha])
    fechas = pd.DataFrame.from_dict(fechas, orient='index').reset_index() #convierte el dict a df
    fechas = fechas.rename(columns={'index':'Fecha', 0:'casos'}) #nombra las columnas
    fechas = fechas.sort_values(by='Fecha') #ordena la columna fecha
    fechas.drop(fechas[fechas.index == 0].index, inplace=True) #elimina fila con nan
    fechas['Fecha'] = pd.to_datetime(fechas['Fecha']) #columna fecha la convierte a datetime
    fechas = fechas.set_index('Fecha', append=False) #columna fecha la convierte a index
    fechas.index = fechas.index.date
    #nuevo df con las fechas completas
    fechas_total = pd.DataFrame(index=np.arange(np.datetime64(str(min(df['FECHA_INGRESO']))), np.datetime64(max(df['FECHA_INGRESO']))))
    fechas_total.index = fechas_total.index.date
    fechas_total = pd.merge(fechas_total,fechas, how='left',left_index=True,right_index=True)
    fechas_total = fechas_total.fillna(0)
    #poly fit
    xaxis = range(len(fechas_total.index))
    coefficients = np.polyfit(xaxis,fechas_total['casos'],npol)
    y_poly = np.poly1d(coefficients)(xaxis).clip(min=0) 
    fechas_total['poly'] = y_poly
    #plot
    fig, ax = plt.subplots()
    plot_date(ax)
    plt.plot(fechas_total.index,fechas_total['casos'], label="real")
    plt.plot(fechas_total.index,y_poly, label="polinomial")
    plt.title(titulo)
    plt.gcf().autofmt_xdate()
    plt.ylabel("No. de casos")
    plt.legend()
    plt.savefig(titulo+'.png', format='png', dpi=1200)
    return fechas_total
#genero df de los 3 tipos de fechas
df_sintomas = casos_nuevos_indiv(titulo="Fecha de sintomas de COVID en Mexico",columna_fecha='FECHA_SINTOMAS',npol=6, estado=False)
df_hosp = casos_nuevos_indiv(titulo="Fecha de hospitalizacion de COVID en Mexico",columna_fecha='FECHA_INGRESO',npol=6, estado=False)
df_def = casos_nuevos_indiv(titulo="Fecha de defuncion de COVID en Mexico",columna_fecha='FECHA_DEF',npol=6, estado=False)
df_sintomas = casos_nuevos_indiv(titulo="Fecha de sintomas de COVID en CDMX",columna_fecha='FECHA_SINTOMAS',npol=6, estado=9)

def casos_nuevos_total(estado, npol, estado_str):
    columnas_fechas = ['FECHA_SINTOMAS', 'FECHA_INGRESO', 'FECHA_DEF']
    list_df = []
    for i, word in enumerate(columnas_fechas):
        list_df.append(casos_nuevos_indiv(titulo= str(word)+" de COVID en "+str(estado_str),columna_fecha=str(word),npol=npol, estado = estado))
    #genera nuevo dataframe con faechas como index
    df_fechas_mex = pd.DataFrame(index=np.arange(np.datetime64(min(df['FECHA_INGRESO'])), np.datetime64(max(df['FECHA_INGRESO']))))
    df_fechas_mex.index = df_fechas_mex.index.date
    #juntar las 3 columnas polinomios de fechas en df_fechas_mex
    total_fechas=[]
    for i in range(3):
        df_fechas_mex = pd.merge(df_fechas_mex,list_df[i].iloc[:,1], how='left',left_index=True,right_index=True)
        total_fechas.append(list_df[i].iloc[:,0].sum()) #guardar el total de las columnas fechas
    #plot
    fig, ax = plt.subplots()
    texto="Total\nsíntomas: "+str(total_fechas[0])+"\ningreso: "+str(total_fechas[1])+"\ndefunción: "+str(total_fechas[2])
    anchored_text = AnchoredText(texto, loc="center left")
    ax.add_artist(anchored_text)
    #################################
    plot_date(ax)
    ax.plot(df_fechas_mex.index,df_fechas_mex.iloc[:,0], label='síntomas')
    ax.plot(df_fechas_mex.index,df_fechas_mex.iloc[:,1], label='hospitalización')
    ax.plot(df_fechas_mex.index,df_fechas_mex.iloc[:,2], label='defunción')
    plt.title("Fechas de COVID en "+str(estado_str))
    plt.gcf().autofmt_xdate()
    plt.ylabel("No. de casos")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fecha de nuevos de COVID en "+str(estado_str)+'.png', format='png', dpi=1200)
    
casos_nuevos_total(estado=False, npol=6, estado_str='México')
casos_nuevos_total(estado=9, npol=6, estado_str='CDMX')
casos_nuevos_total(estado=14, npol=10, estado_str='Jalisco')
casos_nuevos_total(estado=22, npol=8, estado_str='Querétaro')

def casos_acum_indiv(titulo, columna_fecha, npol, estado):
    if estado != False:
        df_aux = df.copy()
        df_aux.drop(df_aux[(df_aux['ENTIDAD_UM'] != estado)].index, inplace = True)
    elif estado == False:
        df_aux = df.copy()
    fechas = Counter(df_aux[columna_fecha])
    fechas = pd.DataFrame.from_dict(fechas, orient='index').reset_index() #convierte el dict a df
    fechas = fechas.rename(columns={'index':'Fecha', 0:'casos'}) #nombra las columnas
    fechas = fechas.sort_values(by='Fecha') #ordena la columna fecha
    fechas.drop(fechas[fechas.index == 0].index, inplace=True) #elimina fila con nan
    fechas['Fecha'] = pd.to_datetime(fechas['Fecha']) #columna fecha la convierte a datetime
    fechas = fechas.set_index('Fecha', append=False) #columna fecha la convierte a index
    fechas.index = fechas.index.date
    #nuevo df con las fechas completas
    fechas_total = pd.DataFrame(index=np.arange(np.datetime64(str(min(df['FECHA_INGRESO']))), np.datetime64(max(df['FECHA_INGRESO']))))
    fechas_total.index = fechas_total.index.date
    fechas_total = pd.merge(fechas_total,fechas, how='left',left_index=True,right_index=True)
    fechas_total = fechas_total.fillna(0)
    #poly fit
    xaxis = range(len(fechas_total.index))
    coefficients = np.polyfit(xaxis,fechas_total['casos'],npol)
    y_poly = np.poly1d(coefficients)(xaxis).clip(min=0) 
    fechas_total['poly'] = y_poly
    #plot
    fig, ax = plt.subplots()
    plot_date(ax)
    plt.plot(fechas_total.index,fechas_total['casos'], label="real")
    plt.plot(fechas_total.index,y_poly, label="polinomial")
    plt.title(titulo)
    plt.gcf().autofmt_xdate()
    plt.ylabel("No. de casos")
    plt.legend()
    plt.savefig(titulo+'.png', format='png', dpi=1200)
    return fechas_total
#%%
#MATRIZ CORRELACION
#hacer columna de hombre y de mujer para reemplazar sexo
df_matcorr = df.drop(['SEXO','ENTIDAD_NAC', 'ENTIDAD_oRES', 'MUNICIPIO_RES', 'OTRO_CASO', 'DIAS_DIF_HOSP'], axis=1)
#df_matcorr = df.loc(['SEXO','ENTIDAD_NAC', 'ENTIDAD_RES', 'MUNICIPIO_RES', 'OTRO_CASO', 'DIAS_DIF_HOSP'], axis=1)
#df_matcorr['hombre'] = df.loc[df['SEXO']== 0]
#df.loc[df['INTUBADO'] == 97, ['INTUBADO']] = 2
corrMatrix = df_matcorr.corr()

corrMatrix.style.background_gradient(cmap='coolwarm')

fig, ax = plt.subplots()

sns.heatmap(corrMatrix, 
            annot=False, 
            xticklabels=range(len(corrMatrix.columns)), 
            yticklabels=1,
            square=True,
            cbar=False)
fig.tight_layout()
fig.savefig('corrmatrix_pos.png', format='png', dpi=1200)
#%%