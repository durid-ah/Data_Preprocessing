import pandas as pd
import requests as req
import io
from sklearn import model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# TODO: Above and beyound point: read directly from url

column_name = ["buying_price", "maint_price", "doors", "passenger_num", "lug_size", "safety", "class"]
response = req.get("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")

level_values = { "low": 1, "med": 2, "high": 3, "vhigh": 4}
numeric_values = { "5more": 5, "more": 5}
sizes = {"small": 1, "med": 2, "big": 3}
class_to_num = { "unacc": 0, "acc": 1, "good": 2, "vgood": 3}

if response.ok:
    data = response.content.decode('utf8')
    car_df = pd.read_csv(io.StringIO(data), names=column_name)
    car_df.replace({"buying_price": level_values,
                "maint_price": level_values,
                "doors": numeric_values,
                "passenger_num": numeric_values,
                "lug_size": sizes,
                "safety": level_values,
                "class": class_to_num
                }, inplace=True)

    car_data = car_df.loc[:, column_name[:-1]]
    car_targets = car_df[car_df.columns[len(car_df.columns)-1]]

    train_data_set, test_data_set, train_target_set, test_target_set = model_selection.train_test_split(car_data,
                                                                                                        car_targets,
                                                                                                        shuffle=True,
                                                                                                        train_size=0.3)

    scaler = StandardScaler()
    scaler.fit(train_data_set)

    train_data_set = scaler.transform(train_data_set)
    test_data_set = scaler.transform(test_data_set)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_data_set, train_target_set)

    y_pred = classifier.predict(test_data_set)
    print("Accuracy:", metrics.accuracy_score(test_target_set, y_pred))

##########################################################################################

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