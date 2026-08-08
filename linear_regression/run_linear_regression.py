import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from linear_regression import LinearRegression

# data prepration (mock up)

X, y = make_regression(
    n_features=6,
    n_samples=500,
    noise=10,
    random_state=42
)

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


if __name__ == "__main__":

    logger.info(f"training data shape : {X_train.shape}")
    logger.info("initializing linear regression")

    model = LinearRegression()

    logger.info("fitting the linear regression model")

    try:
        model.fit(X_train, y_train)
        logger.success("successfully fitted linear regression model")
    except Exception as e:
        logger.error(f"error occured while fitting model : {e}")

    logger.info("saving the model parameters")

    try:
        model.save()
        logger.success("successfully saved model parameters")
    except Exception as e:
        logger.error(f"error occured while saving the model parameters: {e}")

    logger.info("running evaluation on test dataset")

    y_pred = model.predict(X_test)

    logger.info("calculating metrics")

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    logger.info(f"model evaluation metrics: R^2 Score : {r2} | RMSE : {rmse}")

