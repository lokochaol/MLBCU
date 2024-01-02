import pickle

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("データの読み込み")
df = pd.read_csv("train.csv")
df = pd.DataFrame(df)
print(df)

print("===== 欠損値の補完 =====")
print(df.isnull().sum())

print("===== 外れ値の確認 =====")
for name in df:
    if name == "id":
        continue
    df.plot(kind="scatter", x=name, y="Status")
    plt.show()

print("===== データの分割 =====")
le = LabelEncoder()
df["Drug"] = le.fit_transform(df["Drug"])
df["Sex"] = le.fit_transform(df["Sex"])
df["Ascites"] = le.fit_transform(df["Ascites"])
df["Hepatomegaly"] = le.fit_transform(df["Hepatomegaly"])
df["Spiders"] = le.fit_transform(df["Spiders"])
df["Edema"] = le.fit_transform(df["Edema"])
df["Status"] = le.fit_transform(df["Status"])
x = df.loc[:, "N_Days":"Stage"]
t = df["Status"]
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.3, random_state=0)

print("===== 学習テスト =====")
model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, random_state=0)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
print(model.predict(x_test))
probabilities = model.predict_proba(x_test)
print(probabilities)
labels = le.inverse_transform(model.classes_)

print("===== test.csvの予測 =====")
df = pd.read_csv("test.csv")
df = pd.DataFrame(df)
df["Drug"] = le.fit_transform(df["Drug"])
df["Sex"] = le.fit_transform(df["Sex"])
df["Ascites"] = le.fit_transform(df["Ascites"])
df["Hepatomegaly"] = le.fit_transform(df["Hepatomegaly"])
df["Spiders"] = le.fit_transform(df["Spiders"])
df["Edema"] = le.fit_transform(df["Edema"])

x = df.drop("id", axis=1)
probabilities = model.predict_proba(x)
print(probabilities)

print("===== モデルの保存 =====")
print("labels: ", labels)
probabilities_df = pd.DataFrame(probabilities, columns=labels)
print(probabilities_df.head(3))

sub_data = pd.DataFrame(
    {
        "id": df["id"],
        "Status_C": probabilities_df["C"],
        "Status_CL": probabilities_df["CL"],
        "Status_D": probabilities_df["D"],
    }
)
sub_data.to_csv("submission.csv", index=False)
