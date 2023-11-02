
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    obviously_bad_data_filters = [
        df.fare_amount > 0,  # fare_amount in US Dollars
        df.trip_distance <= 100,  # trip_distance in miles
        df.trip_distance > 0,
        df.passenger_count > 0,
        df.tpep_pickup_datetime < df.tpep_dropoff_datetime,
        df.tip_amount >= 0,
        df.tolls_amount >= 0,
        df.improvement_surcharge >= 0,
        df.total_amount >= 0,
        df.congestion_surcharge >= 0,
        df.airport_fee >= 0,
        # TODO: add some logic to filter out what you decide is bad data!
        # TIP: Don't spend too much time on this step for this project though, it practice it is a never-ending process.
    ]

    np.all(obviously_bad_data_filters, axis=0)

    #df = df[np.all(obviously_bad_data_filters, axis=0)]

    return df
