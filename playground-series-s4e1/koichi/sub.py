import pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from autogluon.tabular import TabularDataset, TabularPredictor
from IPython.display import HTML, display
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
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
df.to_csv("conated_data.csv", index=False)
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
model_xg = xgb.XGBClassifier(
    objective="binary:logistic", random_state=0, max_depth=3, n_estimators=160, eta=0.2
)
model_xg.fit(x_train, y_train)
print("score: ", model_xg.score(x_test, y_test))
# cv = GridSearchCV(
#     xgb.XGBClassifier(),
#     {
#         "objective": ["binary:logistic"],
#         "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
#         "n_estimators": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
#         "eta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#     },
#     verbose=2,
# )
# max_depth: 3, n_estimators: 160, eta: 0.2 で最適

# cv.fit(x_train, y_train)
# xg = cv.best_estimator_
# print("cv score: ", xg.score(x_test, y_test), "param: ", xg.get_params())

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
print("score: ", accuracy_score(res, y_test), "param: ", model_li.get_params())

print("===== ランダムフォレスト =====")
model_rf = RandomForestClassifier()
# params = {"n_estimators": [4, 8, 16, 32], "max_depth": [8, 16, 32, 64], "n_jobs": [-1]}
# cv = GridSearchCV(RandomForestClassifier(), params, verbose=2)
# cv.fit(x_train, y_train)
# forest = cv.best_estimator_
# print("cv score: ", forest.score(x_test, y_test))

model_rf.fit(x_train, y_train)
print("score: ", model_rf.score(x_test, y_test), "param: ", model_rf.get_params())

print("===== 相関の小さいデータのdrop =====")
df_2 = df.drop(["HasCrCard"], axis=1)
x = df_2.drop("Exited", axis=1)
t = df_2["Exited"]
x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.3, random_state=0)

print("===== xgboost 2 =====")
model_xg_2 = xgb.XGBClassifier(
    objective="binary:logistic", random_state=0, max_depth=3, n_estimators=160, eta=0.2
)
model_xg_2.fit(x_train, y_train)
print("score: ", model_xg_2.score(x_test, y_test))


print("===== AutoML =====")
# 学習したモデルを保存するディレクトリを指定する。
train_data = df.sample(frac=0.7, random_state=42)
test_data = df.drop(train_data.index)
y_test = test_data["Exited"]
X_test = test_data.drop(columns=["Exited"])
dir_base_name = "agModels"
dir_default = f"{dir_base_name}_default"
predictor = TabularPredictor(label="Exited", path=dir_default)
predictor.fit(train_data=train_data)
y_pred = predictor.predict(X_test)
perf = predictor.evaluate_predictions(
    y_true=y_test, y_pred=y_pred, auxiliary_metrics=True
)
print("Predictions:\n", y_pred)

model_perf = predictor.leaderboard(test_data, silent=True)
display(model_perf)


print("===== 提出予測 =====")
model = predictor
test = pd.read_csv("test.csv")
test = pd.DataFrame(test)
test_id = test["id"]
test = test.drop(["id", "CustomerId", "Surname"], axis=1)
test["Geography"] = le.fit_transform(test["Geography"])
test["Gender"] = le.fit_transform(test["Gender"])
print(test.head(3))
test.to_csv("test_data.csv", index=False)
prediction = model.predict_proba(test).iloc[:, 1]
print(prediction)
output = pd.DataFrame({"id": test_id, "Exited": prediction})
print(output)
output.to_csv("submission.csv", index=False)
