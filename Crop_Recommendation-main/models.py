SEED = 69

import pandas as pd
import json
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import utils
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_models():
    """
    Returns a list of classification models to train
    :Input
        None
    :Returns
        baseModels - a dictionary of classification models
    """
    baseModels = {}
    baseModels['LogisticRegression'] = LogisticRegression(n_jobs = -1)
    baseModels['DecisionTreeClassifier'] = DecisionTreeClassifier()
    baseModels['AdaBoostClassifier'] = AdaBoostClassifier()
    baseModels['GradientBoostingClassifier'] = GradientBoostingClassifier()
    baseModels['RandomForestClassifier'] = RandomForestClassifier(n_jobs = -1, n_estimators=200)
    baseModels['ExtraTreesClassifier'] = ExtraTreesClassifier(n_jobs = -1, n_estimators=200)
    baseModels['SupportVectorClassifier'] = SVC()
    baseModels['KNeighborsClassifier'] = KNeighborsClassifier(n_jobs = -1)
    baseModels['GaussianNB'] = GaussianNB()
    return baseModels


def train_and_evaluate_models(x_train, y_train, x_test, y_test, models):
    """
    Returns a dictionary of trained models
    :Input
        x_train - independant variable to train model on
        y_train - dependant variable
        x_test - independant variable to test the model
        y_test - true labels used to test the model
        models - a ditionary containing objects of classification models
    :Returns
        models - the same dictionary of classification models after training
    """
    print('****Traianing Models****')

    for name, _model in models.items():
        
        _model.fit(x_train, y_train)
        y_pred = _model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(name, score)
    return models

def score_models(x_train, y_train, models, num_folds = 5):
    """
    Cross validates models using sklearn's StratifiedKFold
    :Input
        x_train - independant variable
        y_train - dependant variable
        models -  ditionary containing objects of classification models indexed by name
        num_folds - integer denoting no of folds to use for cv
    :Returns
        scores - dictionary containing mean cross val score of the models
    """
    print('****Cross Validating models****')
    scores = {}
    num_folds = 5

    for name, model in models.items():
        kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle = True)
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy', n_jobs = -1)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        scores[name] = cv_results.mean()

    return scores

def save_models(models, path = "models/"):
    """
    Saves models as pickle files

    :Input
        models - dictionary containing model trained objects

    :Returns:
        None
    """
    print('****Saving Models****')
    for name, model in models.items():
        model_file = path + "{0}.pkl".format(name)

        with open(model_file, 'wb') as file:
            pickle.dump(model, file)
    print('****Saved Successfully****')

#Opening config file to get the dataset path
with open('CONFIG.json') as config_file:
    CONFIG = json.load(config_file)
path = CONFIG['raw_data_path']

#Reading the dataset
data = pd.read_csv(path)

#Getting training and test vaectors
x_train, x_test, y_train, y_test = utils.preprocess_data(data, 0.10)

#Getting models
models = get_models()

#Cross validating the models
scores = score_models(x_train, y_train, models, num_folds = 5)
#Training models
trained_models = train_and_evaluate_models(x_train, y_train, x_test, y_test, models)
print(trained_models['RandomForestClassifier'].predict(np.array([90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362]).reshape(1, -1)))
#Saving models as pickle file
save_models(models)
