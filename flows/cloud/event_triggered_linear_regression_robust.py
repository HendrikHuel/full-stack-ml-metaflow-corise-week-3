from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger, retry, catch, timeout
from metaflow.cards import Markdown, Table, Image, Artifact

URL = "https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@trigger(events=["s3"])
@conda_base(
    libraries={
        "pandas": "2.1.2",  # bump version
        "pyarrow": "13.0.0", # bump version
        #"numpy": "1.21.2",  # omit defining numpy since pandas comes with it
        "scikit-learn": "1.3.2", # bump version
    }
)
class TaxiFarePredictionRobust(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        from utils.preprocessing import clean_data

        df = clean_data(df)

        return df

    @retry(times=3, minutes_between_retries=1)
    @step
    def start(self):
        """Read data seperately, to allow retries."""
        import pandas as pd

        self.df = self.transform_features(pd.read_parquet(self.data_url))

        self.next(self.seperate_features_from_label)
    
    @step
    def seperate_features_from_label(self):

        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values

        self.next(self.linear_model)

    @step
    def linear_model(self):
        "Fit a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

        self.next(self.validate)


    @timeout(minutes=5)
    @card#(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)

        is_even = [0, 1]

        self.next(self.train_models_on_subsets, foreach="is_even")

    @catch
    @timeout(minutes=5)
    @step
    def train_models_on_subsets(self):
        """Train a model for a subset of the data. The subsets could be different districs, for example.
        We try to train a model. In case of failur a hyperthetical inference flow would fall back to the last successfull run. 
        A failure for a subset would be escaleted via internal tools and be degbugged. 
        In the mean time, the parts of the data that did not fail, will profit from the most recent model possible.
        """
        
        import numpy as np
        from sklearn.metrics import mean_absolute_error

        self.mode = self.input
        mask = np.round(self.X, 0) == self.mode

        self.X_mod = self.X[mask]
        self.y_mod = self.y[mask]

        self.model_sub = self.model
        self.model_sub.fit(self.X_mod, self.y_mod)

        self.score = mean_absolute_error(self.y_mod, self.model_sub.predict(self.X_mod))

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Gather segment models.
        """
        import numpy as np

        def score(inp):
            return inp.model_sub, inp.mode, inp.scores

            
        self.results = sorted(map(score, inputs), key=lambda x: -x[2])
        self.next(self.end)
        
    @step
    def end(self):
        """
        End of flow!
        """
        print('Scores:')
        print('\n'.join('%s %f %f' % res for res in self.results))


if __name__ == "__main__":
    TaxiFarePredictionRobust()
