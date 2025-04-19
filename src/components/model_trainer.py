import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as sklearn_r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass 
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join("artifacts", "model.pkl") 

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and import data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # take out the last column
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),  
            }

            param = {
                'RandomForestRegressor': {
                    'n_estimators': [100, 200],  # number of trees
                    'max_depth': [10, 20, 30],    # maximum depth
                    'min_samples_split': [2, 5]   # minimum samples for split
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [100, 150],  # boosting iterations
                    'learning_rate': [0.01, 0.1],  # learning rate
                    'max_depth': [3, 5]            # max depth
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100],   # number of iterations
                    'learning_rate': [0.01, 0.1] # learning rate
                },
                'HistGradientBoostingRegressor': {
                    'max_iter': [100, 150],       # max iterations
                    'learning_rate': [0.01, 0.1]  # learning rate
                },
                'LinearRegression': {
                    'fit_intercept': [True, False]  # whether to include intercept
                },
                'KNeighborsRegressor': {
                    'n_neighbors': [5, 10],       # number of neighbors
                    'weights': ['uniform', 'distance']  # weight options
                },
                'DecisionTreeRegressor': {
                    'max_depth': [10, 20, 30],  # max depth
                    'min_samples_split': [2, 5] # minimum samples for split
                },
                'XGBRegressor': {
                    'n_estimators': [100, 200],  # number of trees
                    'learning_rate': [0.01, 0.1], # learning rate
                    'max_depth': [3, 5]           # max depth
                },
                'CatBoostRegressor': {
                    'iterations': [100, 200],  # number of iterations
                    'learning_rate': [0.01, 0.1],  # learning rate
                    'depth': [5, 10]            # depth
                },
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=param)
            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(
                f"Best model found, {best_model_name} with score: {best_model_score}"
            )

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            # Rename variable to avoid conflict with sklearn's r2_score function
            model_r2_score = sklearn_r2_score(y_test, predicted)
            logging.info(f"R2 score of the model is: {model_r2_score}")
            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)