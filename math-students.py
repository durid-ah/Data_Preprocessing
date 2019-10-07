import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

schools = {"GP": 1, "MS": 2}
sex = {"F": 0, "M": 1}
address = {"U": 1, "R": 0}
fam_size = {"LE3": 1, "GT3": 2}
p_status = {"A": 0, "T": 1}
job = {"at_home": 0, "teacher": 1, "health": 2, "services": 3, "other": 4}
reason = {'home': 1, 'reputation': 2, 'course': 3, 'other': 4}
guardian = {"mother": 1, "father": 2, "other": 3}
yes_no = {"yes": 1, "no": 0}

df = pd.read_csv(filepath_or_buffer="student-mat.csv", sep=";")

# turning the strings into objects
df.replace({
    "school": schools,
    "sex": sex,
    "address": address,
    "famsize": fam_size,
    "Pstatus": p_status,
    "Mjob": job,
    "Fjob": job,
    "reason": reason,
    "guardian": guardian,
    "schoolsup": yes_no,
    "famsup": yes_no,
    "paid": yes_no,
    "activities": yes_no,
    "nursery": yes_no,
    "higher": yes_no,
    "internet": yes_no,
    "romantic": yes_no
}, inplace=True)

# Hot encoding the school, reason and guardian columns
school_column = df.pop('school')
df['GP_school'] = (schools == 1) * 1.0
df['MS_school'] = (schools == 2) * 1.0

reason_column = df.pop('reason')
df['home_reason'] = (reason_column == 1) * 1.0
df['reputation_reason'] = (reason_column == 2) * 1.0
df['course_reason'] = (reason_column == 3) * 1.0
df['other_reason'] = (reason_column == 4) * 1.0

guardian_column = df.pop('guardian')
df['m_guardian'] = (guardian_column == 1) * 1.0
df['f_guardian'] = (guardian_column == 2) * 1.0
df['oth_guardian'] = (guardian_column == 3) * 1.0

# Splitting the labels
data_labels = df.pop('G3')

# creating the test and train sets
train_data_set, test_data_set, train_target_set, test_target_set = train_test_split(df,
                                                                                    data_labels,
                                                                                    shuffle=True,
                                                                                    train_size=0.3)

neighbor_regressor = KNeighborsRegressor(n_neighbors=5)
neighbor_regressor.fit(train_data_set, train_target_set)
predicted = neighbor_regressor.predict(test_data_set)

score = neighbor_regressor.score(test_data_set, test_target_set)
print(score)

