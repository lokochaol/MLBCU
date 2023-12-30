import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
df = pd.DataFrame(df)
print(df.head(3))

print("===== 欠損値の補完 =====")
print("is_null: ", df.isnull().sum())

print("===== 外れ値の確認 =====")
for name in df:
    if name == "id" or name == "Hardness":
        continue
    df.plot(kind="scatter", x=name, y="Hardness")
    plt.show()

print("===== 外れ値の除去 =====")
no = df[(df["allelectrons_Total"] > 8000) & (df["Hardness"] < 6)].index
print("no: ", no)
df3 = df.drop(no)
print("df3.shape: ", df3.shape)

print("===== データの分割 =====")
x = df3.loc[:, "allelectrons_Total":"density_Average"]
t = df3["Hardness"]
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2, random_state=0)

print("===== 学習 =====")
model = LinearRegression()
model.fit(x_train, y_train)
print("score: ", model.score(x_test, y_test))

test_data = pd.read_csv("test.csv")
test_data.head(3)
predictions = model.predict(test_data.drop("id", axis=1))

output = pd.DataFrame({"id": test_data["id"], "Hardness": predictions})
output.to_csv("submission.csv", index=False)
