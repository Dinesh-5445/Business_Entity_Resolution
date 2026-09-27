from sklearn.linear_model import LogisticRegression
import numpy as np

class PlattCalibrator:
    def __init__(self):
        # We use simple Logistic Regression for Platt Scaling
        # C=1.0 is default, we can adjust if needed
        self.model = LogisticRegression(C=1.0, solver='lbfgs')
        self.is_fitted = False
        
    def fit(self, X_scores, y_true):
        """
        Fits the calibrator on out-of-fold scores.
        
        Args:
            X_scores (np.array or list): Raw model scores. Shape (n_samples, 1) or (n_samples,)
            y_true (np.array or list): Ground truth labels (0 or 1).
        """
        X_scores = np.array(X_scores).reshape(-1, 1)
        y_true = np.array(y_true)
        self.model.fit(X_scores, y_true)
        self.is_fitted = True
        
    def predict_proba(self, X_scores):
        """
        Predicts calibrated probabilities.
        
        Args:
            X_scores (np.array or list): Raw model scores.
            
        Returns:
            np.array: Calibrated probabilities for class 1.
        """
        if not self.is_fitted:
            raise ValueError("Calibrator is not fitted.")
        X_scores = np.array(X_scores).reshape(-1, 1)
        # predict_proba returns [prob_class_0, prob_class_1]
        return self.model.predict_proba(X_scores)[:, 1]
