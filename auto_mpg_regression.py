import pandas as pd
import requests as req
import io
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

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

# separate the labels into their own list
mpg_data_targets = mpg_data.pop('mpg')

# split into train and test
train_data_set, test_data_set, train_target_set, test_target_set = model_selection.train_test_split(mpg_data,
                                                                                                    mpg_data_targets,
                                                                                                    shuffle=True,
                                                                                                    train_size=0.3)

scaler = StandardScaler()
scaler.fit(train_data_set)

train_data_set = scaler.transform(train_data_set)
test_data_set = scaler.transform(test_data_set)

neighbor_regressor = KNeighborsRegressor(n_neighbors=5)
neighbor_regressor.fit(train_data_set, train_target_set)
predicted = neighbor_regressor.predict(test_data_set)

score = neighbor_regressor.score(test_data_set, test_target_set)
print(score)
