import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import OneHotEncoder, StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.5417765094895783
exported_pipeline = make_pipeline(
    OneHotEncoder(minimum_fraction=0.1, sparse=False, threshold=10),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=0.05, min_samples_leaf=15, min_samples_split=14, n_estimators=100, subsample=0.05)),
    BernoulliNB(alpha=0.1, fit_prior=False)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
