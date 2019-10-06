import pandas as pd
import requests as req
import io

response = req.get("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
mpg_columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_yr", "origin", "car_name"]

if response.ok:
    data = response.content.decode('utf8')
    # stream the data into a pandas dataframe with the regex "\s+" since the values are separated
    # by multiple spaces rather than the usual commas
    mpg_df = pd.read_csv(io.StringIO(data), names=mpg_columns, sep="\s+")

mpg_data = mpg_df.loc[:, mpg_columns[:-1]]

print(mpg_data)
print(mpg_data.isna().sum())
print(mpg_data.dtypes)
print(mpg_data.horsepower.unique())

mpg_data = mpg_data[mpg_data.horsepower != '?'] # removing unknown information

print('?' in mpg_data.horsepower) # check to see if it worked

# splitting the origin column into categories
origin = mpg_data.pop('origin')
mpg_data["USA"] = (origin == 1) * 1.0
mpg_data["Europe"] = (origin == 2) * 1.0
mpg_data["Japan"] = (origin == 3) * 1.0

