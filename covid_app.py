import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import plotly.express as px
from covid_graficas import casos_nuevos_total
from covid_graficas import casos_acum_total
from covid_graficas import mort_porcentaje

#@st.cache(ttl=3600*24, show_spinner=False)
def load_data():
    df = pd.read_csv("covid_data.csv.zip",low_memory = True)
    df_descrip = pd.read_excel("diccionario_datos_covid19/Descriptores.xlsx")
    df_estados = pd.read_csv("diccionario_datos_covid19/diccionario_estados.csv")
    return df, df_descrip,df_estados

genero_dict = {'Hombre':0,'Mujer':1}
feature_dict = {'No':0,'Si':1}

def get_value(val, my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

def get_key(val, my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def get_fval(val):
    feature_dict = {'No':0,'Si':1}
    for key,value in feature_dict.items():
        if val == key:
            return value

#carga modelo de ml
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def descriptores():
    age = st.number_input("Edad",1,120)
    emb = st.radio("Embarazo",tuple(feature_dict.keys()))
    ren_cron = st.radio("¿Tiene diagnóstico de insuficiencia renal crónica?",tuple(feature_dict.keys()))
    diab = st.radio("¿Tiene un diagnóstico de diabetes?",tuple(feature_dict.keys()))
    inms = st.radio("¿Presenta inmunosupresión?",tuple(feature_dict.keys()))
    epoc = st.radio("¿Tiene un diagnóstico la enfermedad pulmonar obstructiva crónica (EPOC)?",tuple(feature_dict.keys()))
    obes = st.radio("¿Tiene diagnóstico de obesidad?",tuple(feature_dict.keys()))
    otro = st.radio("¿Tuvo contacto con algún otro caso diagnósticado con SARS CoV-2?",tuple(feature_dict.keys()))
    hiper = st.radio("¿Tiene diagnóstico de hipertensión?",tuple(feature_dict.keys()))
    tab = st.radio("¿Tiene hábito de tabaquismo?",tuple(feature_dict.keys()))
    cardio = st.radio("¿Tiene un diagnóstico de enfermedades cardiovasculares?",tuple(feature_dict.keys()))
    asma = st.radio("¿Tiene un diagnóstico de asma?",tuple(feature_dict.keys()))
    sex = st.radio("Sexo",tuple(genero_dict.keys()))
    return age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex

def main():
    df, df_descrip,df_estados = load_data()
    df = df[df.RESULTADO == 1]
    st.title("Análisis y diagnóstico de COVID-19 en México")
    st.header("Una herramienta para visualizar y prevenir") 
    st.sidebar.title("Casos")
    st.sidebar.markdown("Seleccione en el menu una opcion")

    image = Image.open('covid.jpg')
    st.image(image, use_column_width=True)
    ##
    menu = ['README', 'Casos estatales','Hospitalización','Mortalidad antes de hopitalización','Mortalidad hopitalizado','ICU antes del diagnostico de neumonia','ICU despues del diagnostico de neumonia','Ventilador antes de un diagnostico de neumonia y de ICU','Ventilador despues de un diagnostico de neumonia y de ICU']
    submenu = ['Plot', 'Prediction']
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == 'README':
        #st.header("Acerca de este proyecto")
        st.subheader("Análisis y Predicción de Riesgos por COVID-19 en México")
        st.write("Con este proyecto se pretende seguir la evolución y los patrones generados por el COVID-19 mediante gráficas y datos sencillos de visualizar para así generar conciencia de los riesgos que una persona dada pueda llegar a tener tomando en cuenta sus descriptores particulares.")
        st.subheader("Motivación")
        st.write("Proporcionar a la gente con un análisis que evita mostrar conclusiones ambiguas acerca del estado actual del país para asistir una toma de decisiones objetiva tanto por parte de autoridades como de ciudadanos. ")
        st.subheader("Datos actuales")
        st.write("Número de casos positivos de COVID: ", len(df))
        st.write("Número de hospitalizados por COVID: ", df.TIPO_PACIENTE.value_counts()[1])
        st.write("Número de intubados por COVID: ", df.INTUBADO.value_counts()[1])
        st.write("Número de fallecidos por COVID: ", df.BOOL_DEF.value_counts()[1])
        st.write("Número de UCI por COVID: ", df.UCI.value_counts()[1])
        st.subheader("Muestra del dataset y su diccionario de fuentes oficiales")
        st.dataframe(df.head())
        st.dataframe(df_descrip)
        st.image(Image.open('plots/barplot_casos_hos_def.png'), use_column_width=True)
        st.image(Image.open('plots/Casos de COVID en Mexico por rangos de edad.png'), use_column_width=True)    
        st.image(Image.open('plots/Casos de COVID en Mexico por sexo.png') , use_column_width=True)  
        st.image(Image.open('plots/corrmatrix_1.png') , use_column_width=True)    
        st.image(Image.open('plots/sector.png') , use_column_width=True)  
  

        st.write("<hr>", unsafe_allow_html=True)
        st.write(
        "Visita el repositorio en [Github](https://github.com/CT-6282/COVID-19_Paper), "
        "descarga la [Bases de datos Covid-19 en México](https://www.gob.mx/salud/documentos/datos-abiertos-152127), "
        "visita el [Tablero informativo de Covid-19 de México](https://coronavirus.gob.mx/datos/#DOView), "
        "este proyecto fue basado principalmente en este [artículo](https://www.medrxiv.org/content/10.1101/2020.05.03.20089813v1.full.pdf)")
    elif choice == 'Casos estatales':
        st.write(df_estados.iloc[:,[1,2]])
        st.image(Image.open('plots/entidades_casos_pos.png'), use_column_width=True)
        st.image(Image.open('plots/entidades_def_pos.png'), use_column_width=True)
        st.image(Image.open('plots/entidades_let.png'), use_column_width=True)
        dict_estados = dict(zip(list(df_estados.iloc[:,0]),list(df_estados.iloc[:,1])))
        #st.write(dict_estados)
        #st.write(dict_estados.values())
        selected_metrics = st.selectbox(label="Elije un estado...", options=list(dict_estados.values()))
        #st.write(selected_metrics)
        int_estado = get_key(selected_metrics,dict_estados)
        #st.write(int_estado)
        if int_estado == 36:
            ult_sem=casos_nuevos_total(df,estado=False, estado_str='México',show=True)
            st.pyplot()
            casos_acum_total(df,estado=None, estado_str='México',show=True)
            st.pyplot()
            mort_porcentaje(df,estado=None, estado_str='México',show=True)
            st.pyplot()
        else:
            ult_sem = casos_nuevos_total(df,estado=int_estado, estado_str=selected_metrics,show=True)
            st.pyplot()
            casos_acum_total(df,estado=int_estado, estado_str=selected_metrics,show=True)
            st.pyplot()
            mort_porcentaje(df,estado=int_estado, estado_str=selected_metrics,show=True)
            st.pyplot()
        st.write("Suma semanal de casos")
        st.write(ult_sem)
    elif choice == 'Hospitalización':
        st.subheader("Se pretende predecir en base a los descriptores si el paciente contagiado de CoV-2 necesitará hospitalización")
        activity = st.selectbox("Activity", submenu)
        if activity == "Plot":
            st.subheader("Data plot")
            st.image(Image.open('plots/sin_hosp_boxplot.png'), use_column_width=True)
            st.image(Image.open('plots/amb_hosp_casos_pos.png'), use_column_width=True)
            st.image(Image.open('plots/barplot_hospitalizacion_edad.png'), use_column_width=True)
            st.image(Image.open('plots/Tasa de casos de COVID en Mexico por rangos de edad.png'), use_column_width=True)
   
        elif activity=='Prediction':
            st.subheader("Análisis predictivo")
            st.write(pd.read_csv("models/hosp_data_grid_report.csv", index_col=0))
            age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex = descriptores()
            feature_list = [age,get_fval(emb),get_fval(ren_cron),get_fval(diab),get_fval(inms),get_fval(epoc),get_fval(obes),get_fval(otro),get_fval(hiper),get_fval(tab),get_fval(cardio),get_fval(asma),get_value(sex,genero_dict)]
            st.write(feature_list)
            feature_model = np.array(feature_list).reshape(1,-1)
            #ML
            # model_choice = st.selectbox("Seleccione un modelo:",['RF','SVM','LR'])
            # if st.button('Predict'):
            #     if model_choice = 'SVM':-
            model_load = load_model('models/hosp_data_grid.pkl')
            prediction = model_load.predict(feature_model)
            #
            st.write(prediction)
            if prediction == 1:
                st.warning('Necesitara hospitalización')
            else:
                st.success("No necesitara hospitalización")
            # pred_prob = model_load.predict.proba(feature_model)
            # prob_score = {'Ambulante':pred_prob[0][0]*100,'Hospitalizado':pred_prob[0][1]*100}
            # st.json(prob_score)
        ####
    elif choice == 'Mortalidad antes de hopitalización':
        st.subheader("Se pretende predecir en base a los descriptores la mortalidad del paciente contagiado de CoV-2 antes de estar internado en el hospital")
        activity = st.selectbox("Activity", submenu)
        if activity == "Plot":
            st.subheader("Data plot")
            st.image(Image.open('plots/def_pos.png'), use_column_width=True)
            st.image(Image.open('plots/def_edad_histograma.png'), use_column_width=True)
            st.image(Image.open('plots/barplot_defuncion_edad.png'), use_column_width=True)
            st.image(Image.open('plots/Tasa de casos de COVID en Mexico por rangos de edad.png'), use_column_width=True)
            st.image(Image.open('plots/Tasa de letalidad de COVID en México.png'), use_column_width=True)

        elif activity=='Prediction':
            st.subheader("Análisis predictivo")
            st.write(pd.read_csv("models/def_data_grid_report.csv", index_col=0))
            age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex = descriptores()
            feature_list = [age,get_fval(emb),get_fval(ren_cron),get_fval(diab),get_fval(inms),get_fval(epoc),get_fval(obes),get_fval(otro),get_fval(hiper),get_fval(tab),get_fval(cardio),get_fval(asma),get_value(sex,genero_dict)]
            st.write(feature_list)
            feature_model = np.array(feature_list).reshape(1,-1)
            #ML
            # model_choice = st.selectbox("Seleccione un modelo:",['RF','SVM','LR'])
            # if st.button('Predict'):
            #     if model_choice = 'SVM':
            model_load = load_model('models/def_data_grid.pkl')
            prediction = model_load.predict(feature_model)
            #
            st.write(prediction)
            if prediction == 1:
                st.warning('Es probable que fallezca si sus condiciones de salud permanecen igual')
            else:
                st.success("No es probable que fallezca")
    elif choice == 'Mortalidad hopitalizado':
        st.subheader("Se pretende predecir en base a los descriptores la mortalidad del paciente contagiado de CoV-2 al estar internado en el hospital")
        activity = st.selectbox("Activity", submenu)
        if activity == "Plot":
            st.subheader("Data plot")
            st.image(Image.open('plots/def_pos.png'), use_column_width=True)
            st.image(Image.open('plots/def_edad_histograma.png'), use_column_width=True)
            st.image(Image.open('plots/barplot_defuncion_edad.png'), use_column_width=True)
            st.image(Image.open('plots/Tasa de casos de COVID en Mexico por rangos de edad.png'), use_column_width=True)
            st.image(Image.open('plots/Tasa de letalidad de COVID en México.png'), use_column_width=True)

        elif activity=='Prediction':
            st.subheader("Análisis predictivo")
            st.write(pd.read_csv("models/def_hosp_data_grid_report.csv", index_col=0))
            age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex = descriptores()
            uci = st.radio("¿Requirió ingresar a una Unidad de Cuidados Intensivos?",tuple(feature_dict.keys()))
            intub = st.radio("¿Requirió de intubación?",tuple(feature_dict.keys()))
            feature_list = [age,get_fval(emb),get_fval(ren_cron),get_fval(diab),get_fval(inms),get_fval(epoc),get_fval(obes),get_fval(otro),get_fval(hiper),get_fval(tab),get_fval(cardio),get_fval(asma),get_value(sex,genero_dict),get_fval(intub),get_fval(uci)]
            st.write(feature_list)
            #ml
            feature_model = np.array(feature_list).reshape(1,-1)
            model_load = load_model('models/def_hosp_data_grid.pkl')
            prediction = model_load.predict(feature_model)
            st.write(prediction)
            if prediction == 1:
                st.warning('Es probable que fallezca si sus condiciones de salud permanecen igual')
            else:
                st.success("No es probable que fallezca")
    elif choice == 'ICU antes del diagnostico de neumonia':
        st.subheader("Se pretende predecir en base a los descriptores la necesidad de la Unidad de Cuidados Intensivos del paciente contagiado de CoV-2 al estar internado en el hospital sin tener un diagnostico de neumonia")
        activity = st.selectbox("Activity", submenu)
        if activity == "Plot":
            st.subheader("Data plot")
            st.image(Image.open('plots/barplot_casos_uci_int.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por rangos de edad.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por sexo.png'), use_column_width=True)
            st.image(Image.open('plots/Porcentaje de casos de hospitalizacion por COVID en Mexico por rangos de edad.png'), use_column_width=True)

        elif activity=='Prediction':
            st.subheader("Análisis predictivo")
            st.write(pd.read_csv("models/icu_data_grid_report.csv", index_col=0))
            age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex = descriptores()
            feature_list = [age,get_fval(emb),get_fval(ren_cron),get_fval(diab),get_fval(inms),get_fval(epoc),get_fval(obes),get_fval(otro),get_fval(hiper),get_fval(tab),get_fval(cardio),get_fval(asma),get_value(sex,genero_dict)]
            st.write(feature_list)
            #ml
            feature_model = np.array(feature_list).reshape(1,-1)
            model_load = load_model('models/icu_data_grid.pkl')
            prediction = model_load.predict(feature_model)
            st.write(prediction)
            if prediction == 1:
                st.warning('Es probable que necesite ICU')
            else:
                st.success("No es probable que necesite ICU")
    elif choice == 'ICU despues del diagnostico de neumonia':
        st.subheader("Se pretende predecir en base a los descriptores la necesidad de la Unidad de Cuidados Intensivos del paciente contagiado de CoV-2 con un diagnostico de neumonia al estar internado en el hospital")
        activity = st.selectbox("Activity", submenu)
        if activity == "Plot":
            st.image(Image.open('plots/barplot_casos_uci_int.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por rangos de edad.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por sexo.png'), use_column_width=True)
            st.image(Image.open('plots/Porcentaje de casos de hospitalizacion por COVID en Mexico por rangos de edad.png'), use_column_width=True)

        elif activity=='Prediction':
            st.subheader("Análisis predictivo")
            st.write(pd.read_csv("models/icu_neum_data_grid_report.csv", index_col=0))
            age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex = descriptores()
            neum = st.radio("Resultado del diagnostico de neumonía",tuple(feature_dict.keys()))
            feature_list = [age,get_fval(emb),get_fval(ren_cron),get_fval(diab),get_fval(inms),get_fval(epoc),get_fval(obes),get_fval(otro),get_fval(hiper),get_fval(tab),get_fval(cardio),get_fval(asma),get_value(sex,genero_dict),get_fval(neum)]
            st.write(feature_list)
            #ml
            feature_model = np.array(feature_list).reshape(1,-1)
            model_load = load_model('models/icu_neum_data_grid.pkl')
            prediction = model_load.predict(feature_model)
            st.write(prediction)
            if prediction == 1:
                st.warning('Es probable que necesite ICU')
            else:
                st.success("No es probable que necesite ICU")
    elif choice == 'Ventilador antes de un diagnostico de neumonia y de ICU':
        st.subheader("Se pretende predecir en base a los descriptores la necesidad de un ventilador invasivo del paciente contagiado de CoV-2 sin un diagnostico de neumonia y sin haber requerido de ICU al estar internado en el hospital")
        activity = st.selectbox("Activity", submenu)
        if activity == "Plot":
            st.subheader("Data plot")
            st.image(Image.open('plots/hosp_intubados_pos.png'), use_column_width=True)
            st.image(Image.open('plots/barplot_casos_uci_int.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por rangos de edad.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por sexo.png'), use_column_width=True)
            st.image(Image.open('plots/Porcentaje de casos de hospitalizacion por COVID en Mexico por rangos de edad.png'), use_column_width=True)

        elif activity=='Prediction':
            st.subheader("Análisis predictivo")
            st.write(pd.read_csv("models/vent_data_grid_report.csv", index_col=0))
            age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex = descriptores()
            #neum = st.radio("Resultado del diagnostico de neumonía",tuple(feature_dict.keys()))
            feature_list = [age,get_fval(emb),get_fval(ren_cron),get_fval(diab),get_fval(inms),get_fval(epoc),get_fval(obes),get_fval(otro),get_fval(hiper),get_fval(tab),get_fval(cardio),get_fval(asma),get_value(sex,genero_dict)]#,get_fval(neum)]
            st.write(feature_list)
            #ml
            feature_model = np.array(feature_list).reshape(1,-1)
            model_load = load_model('models/vent_data_grid.pkl')
            prediction = model_load.predict(feature_model)
            st.write(prediction)
            if prediction == 1:
                st.warning('Es probable que necesite ventilador')
            else:
                st.success("No es probable que necesite ventilador")
    elif choice == 'Ventilador despues de un diagnostico de neumonia y de ICU':
        st.subheader("Se pretende predecir en base a los descriptores la necesidad de un ventilador invasivo del paciente contagiado de CoV-2 con un diagnostico de neumonia y haber requerido de ICU al estar internado en el hospital")
        activity = st.selectbox("Activity", submenu)
        if activity == "Plot":
            st.subheader("Data plot")
            st.image(Image.open('plots/hosp_intubados_pos.png'), use_column_width=True)
            st.image(Image.open('plots/barplot_casos_uci_int.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por rangos de edad.png'), use_column_width=True)
            st.image(Image.open('plots/Casos de COVID hospitalarios en Mexico por sexo.png'), use_column_width=True)
            st.image(Image.open('plots/Porcentaje de casos de hospitalizacion por COVID en Mexico por rangos de edad.png'), use_column_width=True)
        elif activity=='Prediction':
            st.subheader("Análisis predictivo")
            st.write(pd.read_csv("models/vent_ucineum_data_grid_report.csv", index_col=0))
            age, emb, ren_cron, diab, inms, epoc, obes, otro, hiper, tab, cardio, asma, sex = descriptores()
            neum = st.radio("Resultado del diagnostico de neumonía",tuple(feature_dict.keys()))
            icu = st.radio("¿Requirió ingresar a una unidad de cuidados intensivos?",tuple(feature_dict.keys()))
            feature_list = [age,get_fval(emb),get_fval(ren_cron),get_fval(diab),get_fval(inms),get_fval(epoc),get_fval(obes),get_fval(otro),get_fval(hiper),get_fval(tab),get_fval(cardio),get_fval(asma),get_value(sex,genero_dict),get_fval(icu),get_fval(neum)]
            st.write(feature_list)
            #ml
            feature_model = np.array(feature_list).reshape(1,-1)
            model_load = load_model('models/vent_ucineum_data_grid.pkl')
            prediction = model_load.predict(feature_model)
            st.write(prediction)
            if prediction == 1:
                st.warning('Es probable que necesite ventilador')
            else:
                st.success("No es probable que necesite ventilador")
if __name__ == '__main__':
    main()
