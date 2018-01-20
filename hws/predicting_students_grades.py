# -*- coding: utf-8 -*-

__author__ = 'kirillsavelyev'

# To start in terminal insert "python3 predicting_students_grades.py"

# The task: to predict the student's progress according to his data
# (see columns G1, G2, G3).
# Prototyping can be done in Jupyter notebook,
# the final result must be formalized in the form of a Python module
# Dataset: https://archive.ics.uci.edu/ml/datasets/student+performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost
from sklearn import metrics


def file_reader():
    # Reading files in the DataFrame and replacing string values by categories

    print('Reading Files...')

    # Converting files to data frames
    df1 = pd.read_csv('stud_data/student-mat.csv', sep=';')
    df2 = pd.read_csv('stud_data/student-por.csv', sep=';')

    print('Change values...')
    # Concatenating data frames of students to common data frame
    df = pd.concat([df2, df1], ignore_index=True)

    # Determining string columns of data frame
    cat_columns = df.select_dtypes(include=['object']).columns

    # converting string values to categories
    for c in cat_columns:
        df[c] = df[c].astype('category')

    # saving categories
    categories = {}
    for c in cat_columns:
        categories[c] = [v for v in df[c].cat.categories]

    # replacement of string values by categories
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # Displaying correlation between variables at graph
    try:
        corr_data = df.corr()

        fig = plt.figure(figsize=(33, 33))
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr_data, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, 33, 1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.columns)
        # plt.show()
    except Exception as ex:
        print('Cannot display graph. {}'.format(ex))

    return df


def data_splitter(data):
    # function for splitting data frame to samples
    # input: accepts data frame
    # output: returns four samples X_train, X_test, y_train, y_test

    return list(
        train_test_split(data[0], data[1], test_size=0.33, random_state=42)
    )


def prepare_data(df):
    # Function prepares data for models
    # input: accepts main data frame
    # output: returns samples

    samples = []

    print('Splitting Data Frame...')

    # saving targets
    labels = df[['G1', 'G2', 'G3']]

    # creating data frame for each target
    df_g1 = df.drop(['G1', 'G2', 'G3'], axis=1)
    df_g2 = df.drop(['G2', 'G3'], axis=1)

    # for df_G3 leave only G2 target in data frame,
    # because correlation coefficient between G1 and G2 is high
    df_g3 = df.drop(['G1', 'G3'], axis=1)

    dfs = [df_g1, df_g2, df_g3]

    # sending the data to split, the result to the list
    for i in zip(dfs, [labels[col] for col in labels]):
        samples.append(data_splitter(i))

    return samples


def lr(t_data):
    # LinearRegression - contains the functions of training and predicting
    # input: accepts triple: X_train, X_test, y_train (for G1, G2, G3)
    # output: returns result of prediction (for G1, G2, G3)

    print('Linear Regression')

    def lr_trainer(train):
        # Training function
        # input: accepts couple X_train, y_train
        # output: returns a trained model

        return LinearRegression().fit(train[0], train[1])

    def lr_predictor(model, X_test):
        # Predict function
        # input: accepts a trained model and X_test
        # output: returns result of prediction

        return model.predict(X_test)

    return lr_predictor(lr_trainer(t_data[::2]), t_data[1])


def xgb(t_data):
    # xgboost - contains the functions of training and predicting
    # input: accepts triple: X_train, X_test, y_train (for G1, G2, G3)
    # output: returns result of prediction (for G1, G2, G3)

    print('xgboost Regressor')

    def xgb_trainer(train):
        # Training function
        # input: accepts couple X_train, y_train
        # output: returns a trained model

        return xgboost.XGBRegressor(
            max_depth=3, learning_rate=0.1, n_estimators=100
        ).fit(
            train[0], train[1])

    def xgb_predictor(model, X_test):
        # Predict function
        # input: accepts a trained model and X_test
        # output: returns result of prediction

        return model.predict(X_test)

    return xgb_predictor(xgb_trainer(t_data[::2]), t_data[1])


def teacher(models, split_data):
    # Common function. Model training and forecasting (G1, G2, G3)

    # input:
    # accepts a list of necessary models: ['lr', 'xgb']
    # and list of data: X_train, X_test, y_train, y_test (for G1, G2, G3)

    # data transmitted in the function of models:
    #   X_train, X_test, y_train (G1, G2, G3)

    # output:
    # creates and returns dictionary:
    # { 'LR':   [[y_test_G1, pred_G1],
    #            [y_test_G2, pred_G2],
    #            [y_test_G3, pred_G3]],
    #   'XGBoost': [[y_test_G1, boost_pred_G1],
    #               [y_test_G2, boost_pred_G2],
    #               [y_test_G3, boost_pred_G3]]
    # }

    dict_models = {'lr': lr, 'xgb': xgb}

    dict_results = {}

    print('Teaching:')

    for i in models:
        tl = []
        for j in split_data:
            try:
                tl.append([dict_models[i](j[:3]), j[-1]])
            except KeyError:
                print('No {} model in The System'.format(i))

        dict_results[i] = tl

    return dict_results


def printer(res_dict):
    # Printing  results of prediction  and evaluation of models
    # input: y_test, pred (G1, G2, G3)
    # output: results to console

    # function takes a dictionary:
    # { 'LR':   [[y_test_G1, pred_G1],
    #            [y_test_G2, pred_G2],
    #            [y_test_G3, pred_G3]],
    #   'XGBoost': [[y_test_G1, boost_pred_G1],
    #               [y_test_G2, boost_pred_G2],
    #               [y_test_G3, boost_pred_G3]]
    # }

    for k, v in res_dict.items():
        for j, i in enumerate(v):
            grade = 1 + j
            print('Prediction grades {} with {}: '.format(grade, k),
                  list(i[1]),  # or simple i[1]
                  'Errors of {} prediction Grade {}: '.format(
                      k,
                      grade),
                  '{} {}\n{} {}'.format(
                      metrics.mean_absolute_error.__name__,
                      np.sqrt(metrics.mean_absolute_error(i[0], i[1])),
                      metrics.mean_squared_error.__name__,
                      np.sqrt(metrics.mean_squared_error(i[0], i[1]))
                  ),
                  sep='\n',
                  end='\n\n'
                  )


if __name__ == '__main__':
    # Read and prepare data
    data_samples = prepare_data(file_reader())

    # Teaching of models
    results = teacher(['lr', 'xgb'], data_samples)     # ['lr', 'xgb']

    # Printing predictions and errors
    printer(results)

    # Displaying graph
    plt.show()
