
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    obviously_bad_data_filters = [
        "fare_amount > 0",  # fare_amount in US Dollars
        "trip_distance <= 100",  # trip_distance in miles
        "trip_distance > 0",
        "passenger_count > 0",
        "tpep_pickup_datetime < tpep_dropoff_datetime",
        "tip_amount >= 0",
        "tolls_amount >= 0",
        "improvement_surcharge >= 0",
        "total_amount >= 0",
        "congestion_surcharge >= 0",
        "airport_fee >= 0",
        # TODO: add some logic to filter out what you decide is bad data!
        # TIP: Don't spend too much time on this step for this project though, it practice it is a never-ending process.
    ]

    df = df.query(" & ".join(obviously_bad_data_filters))

    if len(df) == 0:
        raise ValueError("No entries remain after filtering.")

    return df
