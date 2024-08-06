from preprocessData import Preprocessor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import smogn
import torch
import matplotlib.pyplot as plt
import warnings


# export model
def export_model(model, filename):
    """
    Exports model using pickle
    :param model: model to be exported
    :param filename: string of filename to export model to
    :return: none
    """
    pickle.dump(model, open(filename, 'wb'))


def build_models(model, xtrain, ytrain, xtest, ytest):
    """
    Trains and tests given model using given test and training sets; Calculates RMSE.
    :param model: model to train
    :param xtrain: training set of x to use
    :param ytrain: training set of y to use
    :param xtest: test set of x tp use
    :param ytest: test set of y to use
    :return: trained model, rmse score
    """
    # make model
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)

    # evaluate results
    rmse = root_mean_squared_error(ytest, y_pred)
    print("RMSE of non-tuned " + str(model.__class__.__name__) + ": " + str(rmse))

    return model, rmse


def tune_models(model, param_grid, xtrain, ytrain, xtest, ytest, cv=30):
    """
       Trains, tunes, and tests given model using given test and training sets; Calculates RMSE.
       :param model: model to train
       :param xtrain: training set of x to use
       :param ytrain: training set of y to use
       :param xtest: test set of x tp use
       :param ytest: test set of y to use
       :param cv: integer to use for cross validation amount
       :return: tuned model, rmse score
       """
    # apply hyperparameter tuning
    gridsearch = GridSearchCV(model, param_grid=param_grid, cv=cv)
    warnings.filterwarnings("ignore")
    gridsearch.fit(xtrain, ytrain)
    print("Best Parameters of " + str(model.__class__.__name__) + " are: " + str(gridsearch.best_params_))
    hypertuned_model = gridsearch.best_estimator_
    hypertuned_model.fit(xtrain, ytrain)
    y_pred_tuned = hypertuned_model.predict(xtest)

    # evaluate results
    rmse_tuned = root_mean_squared_error(ytest, y_pred_tuned)
    print("RMSE of tuned " + str(model.__class__.__name__) + ": " + str(rmse_tuned))

    return hypertuned_model, rmse_tuned


# file path
file_path = r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\2024_07_08_PVsyst\Compiled Results.xlsx'

# standard random state
rand = 42

# preprocess data
data_processor = Preprocessor(file_path)
display_df, model_df = Preprocessor.process_dataframe(data_processor)

# standardize data
scaler = StandardScaler()
y = model_df['MWh']
X = model_df.drop(['kWh', 'MWh', 'MWh Before Inverter Loss', 'Conversion Loss Difference (MWh)', 'Conversion Loss'],
                  axis=1)
X_scaled = scaler.fit_transform(X)
export_model(scaler, '../res/scaler.pkl')

# split for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=rand)

# make svm model
svr_model, svr_rmse = build_models(SVR(), X_train, y_train, X_test, y_test)

# export
export_model(svr_model, '../res/svr.pkl')

# hyper parameter tuning
svr_param_grid = {'kernel': ('linear', 'rbf', 'sigmoid', 'poly'),
                  'C': [1, 10]}
tuned_svr, tuned_svr_rmse = tune_models(SVR(), svr_param_grid, X_train, y_train, X_test, y_test)

# export
export_model(tuned_svr, '../res/tuned_svr.pkl')

# linear regression model
ridge_model, ridge_rmse = build_models(Ridge(), X_train, y_train, X_test, y_test)

# export
export_model(ridge_model, '../res/ridge.pkl')

# hyper tuned ridge regression
param_ridge = {'solver': ('auto', 'svd', 'lsqr'),
               'alpha': [1, 1.5, 5, 10, 20, 50]}
tuned_ridge, tuned_ridge_rmse = tune_models(Ridge(), param_ridge, X_train, y_train, X_test, y_test)

# export
export_model(tuned_ridge, '../res/tuned_ridge.pkl')

# Lasso model
lasso_model, lasso_rmse = build_models(linear_model.Lasso(), X_train, y_train, X_test, y_test)

# export
export_model(lasso_model, '../res/lasso.pkl')

# hyper tuned lasso regression
param_lasso = {'alpha': [0.01, 0.1, 0.5, 1, 1.5, 5, 10, 20, 50]}
tuned_lasso_model, tuned_lasso_rmse = tune_models(linear_model.Lasso(), param_lasso, X_train, y_train, X_test, y_test)

# export
export_model(tuned_lasso_model, '../res/tuned_lasso.pkl')

# neural network
nn = MLPRegressor(random_state=rand, activation='relu', alpha=10, hidden_layer_sizes=[18, 24, 18],
                  learning_rate='constant', learning_rate_init=0.1, solver='adam')
nn, nn_rmse = build_models(nn, X_train, y_train, X_test, y_test)

# export
export_model(nn, '../res/nn.pkl')

# hyper tuned nn
param_nn = {'solver': ('sgd', 'adam'), 'alpha': [.0001, .001, .01, 1, 5, 10, 20],
            'learning_rate': ('constant', 'adaptive'),
            'learning_rate_init': [.001, .01, .1], 'activation': ('logistic', 'relu'),
            'hidden_layer_sizes': [[10, 20, 20, 10], [20, 50, 20], [10, 15, 20, 20, 15, 10]]}

# this section commented out because it takes long to run and model has already been exported
# tuned_nn_model, tuned_nn_rmse = tune_models(MLPRegressor(random_state=rand), param_nn, X_train, y_train, X_test, y_test)

# export
# export_model(tuned_nn_model, '../res/tuned_nn.pkl')

# multi-task prediction
y_multi = model_df[['MWh', 'MWh Before Inverter Loss']]
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_scaled, y_multi, test_size=0.30, random_state=rand)

multi_nn, multi_nn_rmse = build_models(MultiOutputRegressor(nn), X_train_multi, y_train_multi, X_test_multi,
                                       y_test_multi)

# export
export_model(multi_nn, '../res/multi_nn.pkl')

# multitask lasso
multitask_lasso = linear_model.MultiTaskLasso(alpha=0.1, random_state=rand)
multi_lasso, multi_lasso_rmse = build_models(multitask_lasso, X_train_multi, y_train_multi, X_test_multi,
                                             y_test_multi)
# export
export_model(multi_lasso, '../res/multi_lasso.pkl')

# knn reg
multi_knn, multi_knn_rmse = build_models(KNeighborsRegressor(), X_train_multi, y_train_multi, X_test_multi,
                                         y_test_multi)

# export
export_model(multi_knn, '../res/multi_knn.pkl')

# knn reg tuned
param_knn = {'n_neighbors': [3, 5, 8, 10, 15],
             'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree'],
             'p': [1, 2]}
tuned_multi_knn, tuned_multi_knn_rmse = tune_models(KNeighborsRegressor(), param_knn, X_train_multi, y_train_multi,
                                                    X_test_multi, y_test_multi)

# export
export_model(tuned_multi_knn, '../res/tuned_multi_knn.pkl')

# perform oversampling with SMOTE
'''
train_data = pd.DataFrame(X_train)
train_data['MWh'] = y_train.to_list()
data_smgn = smogn.smoter(data=train_data, y='MWh', under_samp=False)
y_oversampled = data_smgn['MWh']
X_oversampled = data_smgn.drop(['MWh'], axis=1)

# model with oversampling
nn = MLPRegressor(random_state=rand, activation='relu', alpha=10, hidden_layer_sizes=[18, 24, 18],
                  learning_rate='constant', learning_rate_init=0.1, solver='adam')
oversamp_multi_nn, oversamp_multi_nn_rmse = build_models(MultiOutputRegressor(nn), X_oversampled, y_oversampled,
                                                         X_test_multi,
                                                         y_test_multi)

# export
export_model(multi_nn, '../res/oversamp_multi_nn.pkl')
'''
