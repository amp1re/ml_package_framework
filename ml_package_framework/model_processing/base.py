from abc import ABC, abstractmethod

import joblib


class ModelProcessor(ABC):
    """
    Abstract base class for model processing.

    This class provides a template for model processing tasks including fitting,
    evaluating, and saving models. It requires the implementation of fit_model and
    evaluate_model methods in subclasses.

    Methods
    -------
    fit_model(X_train, y_train):
        Abstract method to fit the model with training data.

    evaluate_model(X_valid, y_valid):
        Abstract method to evaluate the model with validation data.

    save_model(filename):
        Save the trained model to a file.

    Attributes
    ----------
    model : sklearn.base.BaseEstimator or similar
        The model instance that will be used for training and evaluation.
    """

    def __init__(self):
        """
        Initialize the ModelProcessor class.

        Sets the model attribute to None. This attribute should be set in the
        subclass once the model is fitted.
        """
        super().__init__()
        self.model = None

    @abstractmethod
    def fit_model(self, X_train, y_train):
        """
        Abstract method to fit the model.
        """
        raise NotImplementedError(
            "The fit_model method must be implemented by subclasses."
        )

    @abstractmethod
    def evaluate_model(self, X_valid, y_valid):
        """
        Abstract method to evaluate the model.
        """
        raise NotImplementedError(
            "The evaluate_model method must be implemented by subclasses."
        )

    def save_model(self, filename):
        """
        Save the trained model to a file.
        """
        if self.model is not None:
            joblib.dump(self.model, filename)
        else:
            print("No model to save.")
