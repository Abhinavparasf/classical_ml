import pickle
from pathlib import Path
import numpy as np

class LinearRegression:
    def __init__(self, epoch=1000, lr=0.01):
        self.epoch = epoch
        self.lr = lr
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0
        for _ in range(self.epoch):
            y_cap = X@w + b
            err = y_cap - y
            dw = 2*X.T@err / X.shape[0]
            db = 2* np.sum(err) / X.shape[0]
            w -= self.lr*dw
            b -= self.lr*db
        self.coef_ = w
        self.intercept_ = b
        return self
    
    def predict(self, X):
        return X@self.coef_ + self.intercept_

    def save(self, path=None):
        if path is None:
            path = Path.cwd()
        model_params = {"coef": self.coef_, "intercept": self.intercept_}
        with open(f"{path}/model_params.pkl", "wb") as file:
            pickle.dump(model_params, file) 

    def load(self, path):
        with open(path, "rb") as file:
            loaded_params = pickle.load(file)
        self.coef_ = loaded_params["coef"]
        self.intercept_ = loaded_params["intercept"]
        return self


    

    

    

