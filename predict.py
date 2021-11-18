from math import sqrt
from google.cloud import storage
import joblib
from TaxiFareModel.params import BUCKET_NAME
from TaxiFareModel.trainer import STORAGE_LOCATION
from TaxiFareModel.data import get_data,clean_data
from TaxiFareModel.utils import split
import ipdb

from sklearn.metrics import mean_absolute_error, mean_squared_error

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"


def get_test_data():
    df = get_data(nrows=1000)
    df = clean_data(df)
    y = df.pop("fare_amount")
    X = df
    X_train, X_test, y_train, y_test = split(X, y)
    return X_train, X_test, y_train, y_test

def get_model():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(STORAGE_LOCATION)
    blob.download_to_filename('model.joblib')
    pipeline = joblib.load('model.joblib')
    return pipeline

def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res

def get_predict(X_test):
    pipeline = get_model()
    #ipdb.set_trace()
    return pipeline.predict(X_test)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_test_data()
    y_pred = get_predict(X_test)
    print(evaluate_model(y_test, y_pred))
