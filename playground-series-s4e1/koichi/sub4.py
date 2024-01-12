import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

df_train = pd.read_csv("train.csv")
df_orig = pd.read_csv("Churn_Modelling.csv")
df_test = pd.read_csv("test.csv")

# Include the original dataset & do some processing
cols = [
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]
df_train = (
    pd.concat((df_train, df_orig), axis=0)
    .drop(["id", "RowNumber"], axis=1)
    .drop_duplicates()
    .dropna()
)
df_train = df_train.loc[df_train.groupby(cols)["Tenure"].idxmax()].drop_duplicates()

train = TabularDataset(df_train)
test = TabularDataset(df_test)

automl = TabularPredictor(label="Exited", problem_type="binary", eval_metric="roc_auc")
automl.fit(train, presets="best_quality")  # "best_quality" に設定することで、不均衡データに対応したモデルを学習する

automl.leaderboard()

prediction = automl.predict_proba(test)

data_submit = pd.read_csv("sample_submission.csv")
data_submit.Exited = prediction[1]
data_submit[["id", "Exited"]].to_csv("simple_ag.csv", index=False)
