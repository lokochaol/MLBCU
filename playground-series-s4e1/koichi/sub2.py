print("===== aws sage maker =====")
import csv
import json

import boto3
import pandas as pd
from sklearn.preprocessing import LabelEncoder

test = pd.read_csv("test.csv")
test = pd.DataFrame(test)
test_id = test["id"]
test = test.drop(["id", "CustomerId", "Surname"], axis=1)

le = LabelEncoder()
test["Geography"] = le.fit_transform(test["Geography"])
test["Gender"] = le.fit_transform(test["Gender"])
csv_data = test.to_csv(index=False)
print(csv_data)

sagemaker_runtime_client = boto3.client("sagemaker-runtime")
response = sagemaker_runtime_client.invoke_endpoint(
    EndpointName="SageMakerEndpoint-2024-1-7-2",
    ContentType="text/csv",
    Body=csv_data,
    Accept="application/json",
)

# 推論結果がJSONで返ってくることを確認する
result = json.loads(response["Body"].read().decode())
values = [i["predicted_label"] for i in result["predictions"]]
print(test_id.shape)
df = pd.DataFrame({"id": test_id, "Exited": values})
print(df.head())
df.to_csv("submission.csv", index=False)
