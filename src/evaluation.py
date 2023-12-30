import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our model
    """

    @abstractmethod
    def calculate_scores(self,y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for model 
        Args:
            y_true: True labels
            y_pred: Predicted labels 
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error 
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculation MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        
class R2(Evaluation):
    """
    Evaluation strategy that uses R2 score
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error 
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("Rmse: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e