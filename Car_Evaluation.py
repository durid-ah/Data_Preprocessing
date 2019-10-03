import pandas as pd
import requests as req
from sklearn import datasets, model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy
import io

# TODO: Above and beyound point: read directly from url

column_name = ["buying_price", "maint_price", "doors", "passenger_num", "lug_size", "safety", "class"]
response = req.get("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")

level_values = { "low": 1, "med": 2, "high": 3, "vhigh": 4}
numeric_values = { "5more": 5, "more": 5}
sizes = {"small": 1, "med": 2, "big": 3}
class_to_num = { "unacc": 0, "acc": 1, "good": 2, "vgood": 3}

if response.ok:
    data = response.content.decode('utf8')
    df = pd.read_csv(io.StringIO(data), names=column_name)
    df.replace({"buying_price": level_values,
                "maint_price": level_values,
                "doors": numeric_values,
                "passenger_num": numeric_values,
                "lug_size": sizes,
                "safety": level_values,
                "class": class_to_num
                }, inplace=True)

    car_data = df.loc[:, column_name[:-1]]
    car_targets = df[df.columns[len(df.columns)-1]]

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
