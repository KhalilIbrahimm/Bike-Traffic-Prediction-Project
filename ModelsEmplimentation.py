import time
import pickle
import sys
import numpy as np
import pandas as pd
from telegram_bot import Bot
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR # Model
from sklearn.neighbors import KNeighborsRegressor # Model
from sklearn.dummy import DummyRegressor # Model
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# The ModelImplementation class contains the implementation of machine learning models,
# evaluation of the models, as well as the process of finding the best model based on cross-validation 
# and evaluating it on the test data at the end.

class ModelImplementation:
    def __init__(self):
        self.final_results = {}
        self.seed = 2023

    def cross_validation(self, X_train, y_train, params, model, model_name):
        """
        Performs cross-validation to find the best hyperparameters for the given model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.
        params (dict): Hyperparameter grid for grid search.
        model (sklearn.base.BaseEstimator): Machine learning model.
        model_name (str): Name of the model.

        Returns:
        Runs cross-validation and updates the results in self.final_results.
        """

        start = time.time()
        print(f"\n** {model_name} **")
        # Define your grid search with time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        gscv = GridSearchCV(estimator=model, param_grid=params, cv=tscv, verbose=3, scoring="neg_root_mean_squared_error", n_jobs=6)
        gscv.fit(X_train, y_train)

        best_model = gscv.best_estimator_
        best_score = np.abs(gscv.best_score_)
        best_parameters = gscv.best_params_
        self.final_results[model_name] = [best_model, best_score]
        end = time.time()
        print(f"\nCross validation best model: {best_model}")
        print(f"With parameters: {best_parameters}")
        print(f"Mean RMSE: {best_score}")
        print(f"Cross Validation time: {((end-start)/60):.3f}m:{end-start}s")

    def DummyRegressor(self, X_train, y_train):
        """
        Executes the Dummy Regressor model with cross-validation to evaluate the model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.

        Returns:
        Runs cross-validation and updates the results in self.final_results.
        """

        params = {"model__strategy":["mean", "median"]}
        dummyregressor_model = DummyRegressor()
        pipeline_model = self.pipeline(dummyregressor_model)
        return self.cross_validation(X_train, y_train, params, pipeline_model, model_name="DummyRegressor")

    def LinearRegression(self, X_train, y_train):
        """
        Executes the Linear Regression model with cross-validation to evaluate the model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.

        Returns:
        Runs cross-validation and updates the results in self.final_results.
        """

        model = LinearRegression()
        params = {}
        pipeline_model = self.pipeline(model)
        return self.cross_validation(X_train, y_train, params, pipeline_model, model_name="LinearRegression")

    def KNeighborsRegressor(self, X_train, y_train):
        """
        Executes the K-Nearest Neighbors Regressor model with cross-validation to evaluate the model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.

        Returns:
        Runs cross-validation and updates the results in self.final_results.
        """

        params = {"model__n_neighbors":[1,3,5,7],"model__weights":["uniform", "distance"]}
        model = KNeighborsRegressor()
        pipeline_model = self.pipeline(model)
        return self.cross_validation(X_train, y_train, params, pipeline_model, model_name="KNeighbiorsRegressor")

    def RandomForestRegressor(self, X_train, y_train):
        """
        Executes the Random Forest Regressor model with cross-validation to evaluate the model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.

        Returns:
        Runs cross-validation and updates the results in self.final_results.
        """

        params = {"model__n_estimators":[10, 100, 200, 1000]}
        model = RandomForestRegressor(random_state=self.seed)
        pipeline_model = self.pipeline(model)
        return self.cross_validation(X_train, y_train, params, pipeline_model, model_name="RandomForestRegressor")

    def SupportVectorRegression(self, X_train, y_train):
        """
        Executes the Support Vector Regression model with cross-validation to evaluate the model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.

        Returns:
        Runs cross-validation and updates the results in self.final_results.
        """

        params = {"model__kernel":["linear", "poly", "sigmoid"], "model__C":[1,3,5], "model__gamma": ["scale", "auto"]}
        svr_model = SVR()
        pipeline_model = self.pipeline(svr_model)
        return self.cross_validation(X_train, y_train, params, pipeline_model, model_name="SupportVectorRegressor")

    def MLPRegressor(self, X_train, y_train):
        """
        Executes the MLPRegressor neural network model with cross-validation to evaluate the model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.

        Returns:
        Runs cross-validation and updates the results in self.final_results.
        """

        params = {"model__hidden_layer_sizes":[(50,), (100,2), (400,4)], "model__activation":["relu"], "model__solver":["adam"], "model__alpha":[0.001, 0.01, 0.1]}
        model = MLPRegressor(random_state=self.seed, max_iter=2000)
        pipeline_model = self.pipeline(model)
        return self.cross_validation(X_train, y_train, params, pipeline_model, model_name="MLPRegressor")

    def find_and_save_best_cross_val_model(self, save_model=None):
        """
        Identifies and saves the best model after cross-validation.

        Args:
        save_model (str): File name to save the best model.

        Returns:
        None
        """

        models_dict = self.final_results
        best_model_info = models_dict[min(models_dict, key=lambda key:np.abs(models_dict[key][1]))]
        print("\n\nOVERVIEW OF ALL MODELS:")
        # Provide an overview of all models with their RMSE
        for model, rmse in models_dict.items():
            print(f"    - Model: {model}, RMSE: {rmse[1]}")

        if save_model:
            with open("model.pkl", "wb") as model_file:
                pickle.dump(best_model_info[0], model_file)
        print("Model saved!")
        print(f"\n\n--> Best Final Cross Val Model: {best_model_info[0]}.\n--> Cross Val Mean RMSE: {np.abs(best_model_info[1])}")
        return f"\n\n--> Best Final Cross Val Model: {best_model_info[0]}.\n--> Cross Val Mean RMSE: {np.abs(best_model_info[1])}"

    def pipeline(self, model):
        """
        Creates a machine learning pipeline with imputation, scaling, and the specified model.

        Args:
        model (sklearn.base.BaseEstimator): The machine learning model to be included in the pipeline.

        Returns:
        sklearn.pipeline.Pipeline: A machine learning pipeline that includes imputation, scaling, and the specified model.
        """

        pipeline_model = Pipeline([("imputer", KNNImputer()), ("scaler", StandardScaler()), ("model", model)])

        return pipeline_model


    def load_model(self):
        """
        Loads the best model from a saved file.

        Returns:
        The best model.
        """

        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        
        return model


    def predict_2023_values(self, df):
        """
        Makes predictions for trafikkmengde in 2023 based on the best model.

        Args:
        df (pd.DataFrame): Data for 2023 for prediction.

        Returns:
        None
        """

        # Selects only data that is from 2023 onwards for prediction.
        data_over_2023 = df[df["År"] >= 2023].drop("Trafikkmengde", axis=1).dropna()
        # Loads the model, which is a pipeline. 
        model = self.load_model()
        # Predicts 2023 trafikkmengde values
        data_over_2023["Trafikkmengde"] = np.round(model.predict(data_over_2023))
        # As the task only wants the date, time, and prediction in the csv file, all other columns are removed. 
        data_over_2023 = data_over_2023.drop(["Globalstraling", "Solskinstid", "Lufttemperatur", "Vindretning", "Vindstyrke", "Lufttrykk", "Vindkast"], axis=1)
        # Saves it in a csv file. 
        data_over_2023.to_csv("predictions.csv")


    def evaluate_final_best_model(self, X_test, y_test):
        """
        Evaluates the final best model on the test data.

        Args:
        X_test (pd.DataFrame): Test data.
        y_test (pd.Series): Target variable for testing.

        Returns:
        None
        """

        model = self.load_model()
        #print("X_test: ", X_test.head())
        y_pred = model.predict(X_test)
        final_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"--> Final Model Test RMSE: {final_test_rmse}")
        return f"--> Final Model Test RMSE: {final_test_rmse}"


    def evaluate_all(self, X_train, y_train, save_model=None):
        """
        Evaluates all models and saves the best model.

        Args:
        X_train (pd.DataFrame): Training data.
        y_train (pd.Series): Target variable for training.
        save_model (str): Filename to save the best model.

        Returns:
        None
        """

        with open('log.txt', 'w') as log_file:
            # Save log output in log-file. 
            sys.stdout = log_file
            self.DummyRegressor(X_train, y_train)
            self.LinearRegression(X_train, y_train)
            self.KNeighborsRegressor(X_train, y_train)
            self.RandomForestRegressor(X_train, y_train)
            self.SupportVectorRegression(X_train, y_train)
            self.MLPRegressor(X_train, y_train)
            self.find_and_save_best_cross_val_model(save_model=save_model)
            sys.stdout = sys.__stdout__
            return self.find_and_save_best_cross_val_model(save_model=save_model)
