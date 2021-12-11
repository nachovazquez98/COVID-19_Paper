from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.compose import make_column_selector
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from pipelinehelper import PipelineHelper
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier 
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier 
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from skopt import BayesSearchCV
'''
robust versions of logistic regression
support vector machines
random forests
gradient boosted decision trees
AdaBoost classifier

scores:
https://towardsdatascience.com/the-3-most-important-composite-classification-metrics-b1f2d886dc7b
https://datascience.stackexchange.com/questions/73974/balanced-accuracy-vs-f1-score
https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
Multiple metric parameter search can be done by setting the scoring parameter to a list 
of metric scorer names or a dict mapping the scorer names to the scorer callables.
The scores of all the scorers are available in the cv_results_ dict at keys ending 
in '_<scorer_name>' ('mean_test_precision', 'rank_test_precision', etc…)
The best_estimator_, best_index_, best_score_ and best_params_ correspond to the 
scorer (key) that is set to the refit attribute.

compare models:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
https://sklearn-evaluation.readthedocs.io/en/stable/user_guide/grid_search.html
https://www.kaggle.com/crawford/hyperparameter-search-comparison-grid-vs-random
https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

#importante
#https://dl.acm.org/doi/10.5555/2503308.2188395
https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
https://scikit-learn.org/0.15/modules/model_evaluation.html
https://scikit-learn.org/stable/modules/model_evaluation.html
https://amueller.github.io/aml/04-model-evaluation/10-evaluation-metrics.html
https://scikit-learn.org/dev/auto_examples/inspection/plot_permutation_importance_multicollinear.html
https://imbalanced-learn.org/stable/references/ensemble.html
macro average recall = balanced accuracy 
macro average precision
#bayes
https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html#skopt.BayesSearchCV
https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/
https://towardsdatascience.com/hyperparameter-optimization-with-scikit-learn-scikit-opt-and-keras-f13367f3e796
https://github.com/RMichae1/PyroStudies/blob/master/Bayesian_Optimization.ipynb
https://medium.datadriveninvestor.com/alternative-hyperparameter-optimization-techniques-you-need-to-know-part-2-e9b0d4d080a9
https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html

probar metrica 
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177678
'''
def Gridsearchcv(X_train, X_test, y_train, y_test):
    ############
    # Scale numeric values
    num_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())])
    
    preprocessor = ColumnTransformer(
        remainder='passthrough',
        transformers=[
            ('num', num_transformer, make_column_selector(pattern='EDAD'))
            ])
    ############
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        # ('ros', RandomOverSampler(sampling_strategy='all')),
        # ('r', SMOTEENN(
        #     smote = SMOTE(),
        #     enn = EditedNearestNeighbours())),#majority all not minority
        # ('smote', SMOTE()),
        # ('smoten', SMOTEN()), #only for categorical features
        ('clf', PipelineHelper([
            ('svc', SVC()),
            ('gb', GradientBoostingClassifier()),
            ('xgb', XGBClassifier(use_label_encoder=False)),
            ('eec', EasyEnsembleClassifier()),
            ('rbc', RUSBoostClassifier()),
            ('bbc', BalancedBaggingClassifier()),
            ('brf', BalancedRandomForestClassifier()),
        ])),
    ])

    params = {
    'clf__selected_model': pipe.named_steps['clf'].generate({

        # # #EasyEnsembleClassifier
        'eec__n_estimators' : [10, 25, 50, 100],
        'eec__warm_start' : [False, True],
        'eec__replacement' : [False, True],

        # # #RUSBoostClassifier
        'rbc__algorithm' : ['SAMME','SAMME.R'],
        'rbc__n_estimators' : [10, 50, 100, 200, 500],
        'rbc__learning_rate' : [1e-3, 1e-2, 1e-1, 0.5, 1.],
        
        # # #BalancedBaggingClassifier
        'bbc__base_estimator': [HistGradientBoostingClassifier(), None],
        'bbc__n_estimators' : [10, 20, 50, 100],
        'bbc__max_samples':[0.5,0.6,0.7,0.8,0.9,1.0],
        'bbc__max_features':[0.5,0.6,0.7,0.8,0.9,1.0],

        # #BalancedRandomForestClassifier
        'brf__criterion': ['gini', 'entropy'],
        'brf__n_estimators' : [int(x) for x in np.linspace(start = 10, stop = 1000, num = 4)],
        'brf__max_depth' : [None, 7, 15, 45],
        'brf__min_samples_split' : [int(x) for x in np.linspace(start = 2, stop = 20, num = 3)],
        'brf__min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 15, num = 3)],
        'brf__max_features' : ['auto', 'sqrt', 'log2'],
        'brf__class_weight' : ['balanced', 'balanced_subsample'],

        # # #svm 
        'svc__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000], 
        'svc__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50],
        'svc__kernel': ['rbf'],
        
        # # #gb 3780
        "gb__learning_rate": [0.001, 0.01, 0.1],
        "gb__max_depth":[3,7,9],
        "gb__max_features":["log2","sqrt"],
        "gb__subsample":[0.5, 0.7, 1.0],
        "gb__n_estimators":[10, 100, 1000],
        
        # #xgboost
        'xgb__learning_rate' : [1e-3, 1e-2, 1e-1, 0.5, 1.],  
        'xgb__min_child_weight': np.arange(1, 21, 5),
        'xgb__subsample': np.arange(0.05, 1.01, 0.05),
        'xgb__verbosity': [0],

        # 'xgb__booster': ['gbtree', 'gblinear' ,'dart'], 
        # 'xgb__learning_rate' : [1e-3, 1e-2, 1e-1, 0.5, 1.], 
        # 'xgb__min_child_weight': range(1, 21, 5),
        # 'xgb__subsample': np.arange(0.05, 1.01, 0.05),
        # 'xgb__max_depth': [15,20,25],
        # 'xgb__verbosity': [0],

        # 'xgb__n_estimators': [100],
        # 'xgb__max_depth': range(1, 11),
        # 'xgb__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        # 'xgb__subsample': np.arange(0.05, 1.01, 0.05),
        # 'xgb__min_child_weight': range(1, 21),
        # 'xgb__verbosity': [0], # add this line to slient warning 
        
        # 'xgb__n_estimators': [400, 700, 1000],
        # 'xgb__colsample_bytree': [0.7, 0.8],
        # 'xgb__max_depth': [15,20,25],
        # 'xgb__reg_alpha': [1.1, 1.2, 1.3],
        # 'xgb__reg_lambda': [1.1, 1.2, 1.3],
        # 'xgb__subsample': [0.7, 0.8, 0.9],
        # 'xgb__eval_metric' : ['mlogloss']
        }),
    }
    scoring = {
        'ba': 'balanced_accuracy',
        'acc': make_scorer(accuracy_score),
        'ap': 'average_precision', 
        'F1' : 'f1', 
        'ra': 'roc_auc', 
        'rc': 'recall'
        }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    #cv = RepeatedStratifiedKFold(n_splits =10, n_repeats=5)
    #https://towardsdatascience.com/hyper-parameter-tuning-with-randomised-grid-search-54f865d27926
    #n_iter: 30,60, 100
    grid = RandomizedSearchCV(
        pipe, 
        params,
        refit = 'ra',
        cv = cv, 
        verbose = 3, 
        n_jobs=-1,
        n_iter = 60,
        scoring= scoring,
        return_train_score = True
        )

    grid.fit(X_train, y_train)
    df_grid=pd.DataFrame(grid.cv_results_)
    df_grid = df_grid.sort_values(by=['mean_test_ba'],ascending=False)
    df_grid = df_grid[[
        'param_clf__selected_model',
        'params',
        'mean_fit_time',
        'std_fit_time',
        'mean_test_ba',
        'std_test_ba',
        'rank_test_ba',
        'mean_test_ap',
        'std_test_ap',
        'rank_test_ap',
        'mean_test_ra',
        'std_test_ra',
        'rank_test_ra',
        'mean_test_F1', 
        'std_test_F1', 
        'rank_test_F1'
        'mean_test_acc', 
        'std_test_acc', 
        'rank_test_acc'
    ]]

    print("Best-Fit Parameters From Training Data:\n",grid.best_params_)
    grid_predictions = grid.best_estimator_.predict(X_test) 
    report = classification_report(y_test, grid_predictions, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(confusion_matrix(y_test, grid_predictions))

    return grid, df_grid, report
