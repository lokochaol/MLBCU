# Pythonの公式イメージをベースに使用
FROM python:3.11

# 作業ディレクトリを設定
WORKDIR /workspace

# 必要なPythonライブラリをインストール
# numpyとpandasはscikit-learnでよく使われるので含めています
# matplotlibとseabornはデータの可視化に便利
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

