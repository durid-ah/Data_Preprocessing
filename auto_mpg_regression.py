import pandas as pd
import requests as req
import io

response = req.get("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
mpg_columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_yr", "origin", "car_name"]

if response.ok:
    data = response.content.decode('utf8')
    mpg_df = pd.read_csv(io.StringIO(data), names=mpg_columns, sep="\s+")

mpg_data = mpg_df.loc[:, mpg_columns[:-1]]

print(mpg_data)
print(mpg_data.isna().sum())
print(mpg_data.dtypes)
print(mpg_data.horsepower.unique())

mpg_data = mpg_data[mpg_data.horsepower != '?'] # removing unknown information

print('?' in mpg_data.horsepower) # check to see if it worked
