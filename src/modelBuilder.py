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
import warnings


# export model
def export_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def build_models(model, xtrain, ytrain, xtest, ytest):
    # make model
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)

    # evaluate results
    rmse = root_mean_squared_error(ytest, y_pred)
    print("RMSE of non-tuned " + str(model.__class__.__name__) + ": " + str(rmse))

    return model, rmse


def tune_models(model, param_grid, xtrain, ytrain, xtest, ytest, cv=30):
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
#tuned_nn_model, tuned_nn_rmse = tune_models(MLPRegressor(random_state=rand), param_nn, X_train, y_train, X_test, y_test)

# export
#export_model(tuned_nn_model, '../res/tuned_nn.pkl')

# multi-task prediction
y_multi = model_df[['MWh', 'MWh Before Inverter Loss']]
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_scaled, y_multi, test_size=0.30, random_state=rand)

multi_nn, multi_nn_rmse = build_models(MultiOutputRegressor(nn), X_train_multi, y_train_multi, X_test_multi,
                                       y_test_multi)

# export
export_model(multi_nn, '../res/multi_nn.pkl')
