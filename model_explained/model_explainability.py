# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
'''
scores
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177678
https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/

'''


# %%
import os, sys
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
from sklearn import metrics
import joblib as joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier 
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier 
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVC
import shap
from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.classifier import confusion_matrix
from yellowbrick.features import pca_decomposition
from yellowbrick.classifier import classification_report
# %%
path = "/home/nacho/Documents/coronavirus/COVID-19_Paper/"
os.chdir(os.path.join(path)) 
#%%
def pred_label(filename): 
    if filename.find('df_caso0') != -1:
        label = 'hosp_critica'
    if filename.find('caso1') != -1:
        label = 'TIPO_PACIENTE'
    if filename.find('caso2') != -1 or filename.find('caso3') != -1 or filename.find('df_caso_3_1') != -1 or filename.find('df_caso_3_2') != -1 or filename.find('df_caso_3_3') != -1:
        label = 'BOOL_DEF'
    if filename.find('caso5') != -1 or filename.find('df_caso5_1') != -1:
        label = 'UCI'
    if filename.find('caso6') != -1 or filename.find('caso7') != -1 or filename.find('df_caso_7_1') != -1 or filename.find('df_caso_7_2') != -1 or filename.find('df_caso_7_3') != -1:
        label = 'INTUBADO'
    return label
#%%
dict_clf = {
  "gb": GradientBoostingClassifier(),
  "xgb": XGBClassifier(use_label_encoder=False),
  "eec": EasyEnsembleClassifier(),
  "svc": SVC(),
  "rbc": RUSBoostClassifier(),
  "bbc": BalancedBaggingClassifier(),
  "brf": BalancedRandomForestClassifier()
  }
#%%
for subdir, dirs, files in os.walk('models/all_data/20_80'):
    for file in files:
        if file.endswith(".pkl"):
            file_path = subdir + "/" + file
            file_name = file.split('.', 1)[0]
            print(file_name)
            df_data = pd.read_csv("prediction_data/completed_datasets/"+re.sub(r'\D+$', '', file_name)+str(".zip"))
            #df_data = df_data.sample(frac=data_percentage)
            #separar datos
            label = pred_label(file_name)
            print(label)
            X = df_data.loc[:, df_data.columns != label]
            y = df_data.loc[:, label]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,stratify=y,shuffle=True)
            print(y.value_counts())       
            grid = joblib.load(file_path)
            grid_report = pd.read_csv(subdir + "/" + file_name+"_report.csv", index_col=0)
            #report
            df_grid = pd.DataFrame(grid.cv_results_).sort_values(by=['mean_test_ba'],ascending=False)
            df_grid=df_grid.sort_values(by=['mean_test_ba'],ascending=False)
            df_grid=df_grid[[
                'param_clf__selected_model',
                'params',
                'rank_test_ba',
                'mean_fit_time',
                'mean_test_ba',
                'mean_test_F1',
                'mean_test_ra',
                'mean_test_rc',
                'mean_test_ap'
                ]]
            display(df_grid)
            clf = grid.best_params_['clf__selected_model'][0]
            clf_params = grid.best_params_['clf__selected_model'][1]
            clf_params = {clf+"__"+k: v for k, v in clf_params.items()}
            #model
            num_transformer=Pipeline(steps=[
                ('scaler',MinMaxScaler())])
            preprocessor=ColumnTransformer(
                remainder='passthrough',
                transformers=[('num',num_transformer,['EDAD'])])
            model=Pipeline([
                ('preprocessor',preprocessor),
                #('r', SMOTEENN(smote = SMOTE(), enn = EditedNearestNeighbours())),
                (clf, dict_clf[clf])
                ])
            #train
            model.set_params(**clf_params)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            print(classification_report(y_test,y_pred))
            metrics.roc_auc_score(y_test, y_pred)
            metrics.balanced_accuracy_score(y_test, y_pred)
            metrics.average_precision_score(y_test, y_pred)
            metrics.f1_score(y_test, y_pred)
            #se preprocesan los datos
            X_train_scaled = pd.DataFrame(model.named_steps['preprocessor'].fit_transform(X_train),columns = X_train.columns)
            X_test_scaled = pd.DataFrame(model.named_steps['preprocessor'].fit_transform(X_test),columns = X_test.columns)
            #cambia el orden de edad y sexo
            X_train_scaled[['EDAD','SEXO']]=X_train_scaled[['SEXO','EDAD']]
            X_test_scaled[['EDAD','SEXO']]=X_test_scaled[['SEXO','EDAD']]
            #prueba train
            X_test_scaled.iloc[20, :]
            y_test.iloc[20]
            model.named_steps[clf].predict(X_test_scaled.iloc[20, :].values.reshape(1, -1))
            model.named_steps[clf].predict_proba(X_test_scaled.iloc[20, :].values.reshape(1, -1))
            # use Kernel SHAP to explain test set predictions
            # num_features = len(X_test.columns)
            # explainer = shap.KernelExplainer(model = model.named_steps[clf].predict_proba, data = X_train_scaled, link = 'logit')
            # shap_values=explainer.shap_values(X=X_test_scaled,nsamples=30,l1_reg="num_features(12)")
            # shap.summary_plot(shap_values,X_test)
            # shap.summary_plot(shap_values[1],X_test_scaled
            
            #roc auc curve
            roc_auc(model, X_train, y_train, X_test=X_test, y_test=y_test, classes=['NO '+label, label])
            confusion_matrix(model,X_train,y_train, X_test, y_test, classes=['NO '+label, label])
            pca_decomposition(X, y, scale=True, classes=['NO '+label, label])
            pca_decomposition(X, y, scale=True, classes=['NO '+label, label], projection=3)
            visualizer = classification_report(model, X_train, y_train, X_test, y_test, classes=['NO '+label, label], support=True)
