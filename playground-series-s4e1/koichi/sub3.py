import json

import boto3
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# データを読み込んでDataFrameを作成する例（必要に応じて変更してください）
test = pd.read_csv("test.csv")
test = pd.DataFrame(test)
test_id = test["id"]
test = test.drop(["id", "CustomerId", "Surname"], axis=1)

# Label Encodingなどの前処理を適用する例（必要に応じて変更してください）
le = LabelEncoder()
test["Geography"] = le.fit_transform(test["Geography"])
test["Gender"] = le.fit_transform(test["Gender"])

# データをバッチに分割してリクエストを送信する
chunk_size = 5000  # バッチサイズ
num_chunks = len(test) // chunk_size + 1

sagemaker_runtime_client = boto3.client("sagemaker-runtime")

results = []
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(test))
    chunk_data = test.iloc[start_idx:end_idx].to_csv(index=False, header=False)

    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName="SageMakerEndpoint-2024-1-7",
        ContentType="text/csv",
        Body=chunk_data,
        Accept="application/json",
    )

    result = json.loads(response["Body"].read().decode())
    results.extend(result["predictions"])

print(result["predictions"])
# 推論結果を処理してCSVファイルに保存する
values = [eval(item["probabilities"])[1] for item in results[: len(test)]]
df = pd.DataFrame({"id": test_id, "Exited": values})
df.to_csv("submission.csv", index=False)
