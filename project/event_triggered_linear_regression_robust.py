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
class TaxiFarePrediction(FlowSpec):
    data_url = Parameter("data_url", default=URL)

    @retry(times=3, minutes_between_retries=1)
    @step
    def start(self):
        """Read data seperately to allow retries."""
        import pandas as pd

        self.df = pd.read_parquet(self.data_url)

        self.next(self.transform_features)
    
    @step
    def transform_features(self):
        """Clean data."""
        from utils.preprocessing import clean_data

        self.df = clean_data(self.df)

        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values

        self.next(self.linear_model)

    @step
    def linear_model(self):
        "Import model."
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

        self.next(self.validate)


    @timeout(minutes=5)
    @step
    def validate(self):
        """Validate model."""

        from sklearn.model_selection import cross_val_score

        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)

        self.is_even = [0, 1]

        self.next(self.train_models_on_subsets, foreach="is_even")

    @catch(var="train_of_seg_model_failed")
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
        self._name = f"Segment_{self.mode}"
        mask = np.round(self.X, 0) == self.mode

        self.X_mod = self.X[mask].reshape(-1, 1)
        self.y_mod = self.y[mask.flatten()]

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
            if not inp.train_of_seg_model_failed:
                return inp._name, inp.mode, inp.score
            else: 
                return inp._name, inp.mode, -999
            
        self.results = sorted([score(inp) for inp in inputs], key=lambda x: -x[-1])
        self.next(self.end)
        
    @step
    def end(self):
        """
        End of flow!
        """
        print('Scores:')
        print('\n'.join(["{}: Mode {:.0f} - MAE {:.2f}".format(*res) for res in self.results]))


if __name__ == "__main__":
    TaxiFarePrediction()
