from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier 
from pipelinehelper import PipelineHelper
from sklearn.linear_model import LogisticRegression
from pactools.grid_search import GridSearchCVProgressBar
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
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
        ('clf', PipelineHelper([
            #('SupVM', SVC())
            #('rf', RandomForestClassifier())
            #('lr', LogisticRegression(solver='saga'))
            #('gb', GradientBoostingClassifier())
            ('xgb', XGBClassifier())
        ])),
    ])

    params = {
    'clf__selected_model': pipe.named_steps['clf'].generate({
        
        # 'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100],
        # 'SupVM__gamma' : [0.0001, 0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50, 75, 100] ,
        # 'SupVM__kernel': ['rbf']
        
        # 'lr__penalty' : ['l1', 'l2'],
        # 'lr__C' : np.logspace(-4, 4, 20),
        # 'lr__tol' : [1e-4,1e-5],
        # 'lr__max_iter' : [350]

        #randmon
        # 'rf__n_estimators' : [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)],
        # 'rf__max_features' : ['auto', 'sqrt', 'log2],
        # 'rf__max_depth' : [int(x) for x in np.linspace(1, 45, num = 3)],
        # 'rf__min_samples_split' : [5, 10]

        #gb
        # "gb__learning_rate": [0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        # "gb__max_depth":[3,7,8,9,10,50],
        # "gb__max_features":["log2","sqrt"],
        # "gb__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
        # "gb__n_estimators":[10, 50, 100, 200, 500]
        
        #xgboost
        # 'xgb__n_estimators': [100],
        # 'xgb__max_depth': range(1, 11),
        # 'xgb__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        # 'xgb__subsample': np.arange(0.05, 1.01, 0.05),
        # 'xgb__min_child_weight': range(1, 21),
        # 'xgb__verbosity': [0] # add this line to slient warning message
        }),
    }
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)#*
    #grid = GridSearchCV(pipe, params,refit = 'f1',cv = cv, verbose = 3, n_jobs=-1,scoring= ['f1','balanced_accuracy']) #'balanced_accuracy'
    grid = RandomizedSearchCV(pipe, params,refit = 'balanced_accuracy',cv = cv, verbose = 3, n_jobs=-1,scoring= ['f1','balanced_accuracy']) #'balanced_accuracy'
    grid.fit(X_train, y_train)
    print("Best-Fit Parameters From Training Data:\n",grid.best_params_)
    grid_predictions = grid.best_estimator_.predict(X_test) 
    report = classification_report(y_test, grid_predictions, output_dict=True)
    report = pd.DataFrame(report).transpose()
    print(report)
    print(confusion_matrix(y_test, grid_predictions))
    return grid, report