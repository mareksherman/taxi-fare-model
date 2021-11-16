from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.utils import split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
        return compute_rmse(y_pred, y_test)


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
    print(trainer.evaluate(X_test,y_test))
