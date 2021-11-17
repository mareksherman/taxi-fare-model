from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.utils import split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from termcolor import colored

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[DE] [Munich] [mareksherman] lin reg v.01"

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
                            ('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())
                            ])
        time_pipe = Pipeline([
                            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))
                            ])
        preproc_pipe = ColumnTransformer([
                                        ('distance', dist_pipe,
                                         ["pickup_latitude", "pickup_longitude",
                                          'dropoff_latitude', 'dropoff_longitude']),
                                        ('time', time_pipe, ['pickup_datetime'])
                                        ], remainder="drop")
        self.pipeline = Pipeline([
                        ('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())
                        ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse,2)

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    df = get_data(nrows=1000)
    df = clean_data(df)
    y = df.pop("fare_amount")
    X = df
    X_train, X_test, y_train, y_test = split(X,y)
    dist_trans = DistanceTransformer()
    distance = dist_trans.fit_transform(X_train, y_train)
    time_enc = TimeFeaturesEncoder('pickup_datetime')
    time_features = time_enc.fit_transform(X_train, y_train)
    trainer = Trainer(X_train,y_train)
    trainer.run()
    rsme = trainer.evaluate(X_test,y_test)
    print(rsme)
    trainer.save_model()
    experiment_id = trainer.mlflow_experiment_id

    print(
    f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
