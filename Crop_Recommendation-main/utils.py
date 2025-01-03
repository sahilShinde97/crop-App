#Importing required libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def preprocess_data(dframe, _test_size = 0.10):
    """
    Preprocesses a given data frame.
    Preprocessing includes scaling using sklearn's MinMaxScaler

    :Input
        dframe - a pandas data frame to process
        _test_size - a float denoting size of the testing set
    :Returns:
        (x_train_scaled, x_test_scaled, y_train, y_test) - a tuple containing training and test vectors after Preprocessing
    """

    x, y = dframe.drop('label', axis = 'columns'), dframe.label
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = _test_size, random_state = 0)

    return (x_train, x_test, y_train, y_test)


def generate_seaborn_plots(dframe, colname):

    crops_group = dframe.groupby('label')
    fig, ax = plt.subplots()
    fig = plt.figure(figsize = (15, 3))
    plt.xticks(rotation = 90)
    plt.ylabel(colname)
    ax = sns.barplot(crops_group[colname].mean().index, crops_group[colname].mean().values)
    return fig

@st.cache(allow_output_mutation = True)
def load_models():
        """
        Loads models from the specified in the CONIFG file
        :Input
            model_path - string denotingpath to load models from

        :Returns:
            models - a dictionary containing loaded models
        """
        print('****Loading Models****')
        models = {}
        
        with open('CONFIG.json') as f:
            config = json.load(f)

        model_file_paths = os.listdir(config["model_folder"])

        for file in model_file_paths:
            with open(config["model_folder"] + file, 'rb') as File:
                models[file] = pickle.load(File)
        print('****Loaded****')
        #print(models)
        return models
