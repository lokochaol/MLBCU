import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("データの読み込み")
df = pd.read_csv("train.csv")
df = pd.DataFrame(df)
df = df.drop("id", axis=1)
print(df, df.shape)
original = pd.read_csv("Churn_Modelling.csv")
original = pd.DataFrame(original)
original = original.drop("RowNumber", axis=1)
print(original, original.shape)

df = pd.concat([df, original], ignore_index=True)
print(df, df.shape)

print("===== データの前処理=====")
df = df.dropna()
print(df.isnull().sum())
df = df.drop(["Surname", "CustomerId"], axis=1)
print(df.head(3))
le = LabelEncoder()
df["Geography"] = le.fit_transform(df["Geography"])
df["Gender"] = le.fit_transform(df["Gender"])
print(df.head(3))

print("===== 相関 =====")
sns.clustermap(df.corr())
plt.show()

print("===== データの分割 =====")
x = df.drop("Exited", axis=1)
t = df["Exited"]
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.3, random_state=0)
print(
    "x_train: ",
    x_train.shape,
    "x_test: ",
    x_test.shape,
    "y_train: ",
    y_train.shape,
    "y_test: ",
    y_test.shape,
)

print("===== xgboost =====")
model_xg = xgb.XGBClassifier(objective="multi:softmax", num_class=2, random_state=0)
model_xg.fit(x_train, y_train)
print("score: ", model_xg.score(x_test, y_test))

print("===== 重回帰分析 =====")
model_li = LinearRegression()
model_li.fit(x_train, y_train)
res = model_li.predict(x_test)
for i in range(len(res)):
    if res[i] > 0.44:
        res[i] = 1
    else:
        res[i] = 0
print(res)
print("score: ", accuracy_score(res, y_test))

print("===== 試しに関係なさそうな変数を減らしてみる =====")


print("===== 提出予測 =====")
test = pd.read_csv("test.csv")
test = pd.DataFrame(test)
test_id = test["id"]
test = test.drop(["id", "CustomerId", "Surname"], axis=1)
test["Geography"] = le.fit_transform(test["Geography"])
test["Gender"] = le.fit_transform(test["Gender"])
print(test.head(3))
prediction = model_xg.predict(test)
output = pd.DataFrame({"id": test_id, "Exited": prediction})
print(output)
output.to_csv("submission.csv", index=False)
