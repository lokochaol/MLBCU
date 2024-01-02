# MLBCU

## makeコマンド
### 初期設定

venvに入り
```
pip install -r requirements.txt
```
を実行。


https://www.kaggle.com/settings/account にアクセスし、
'Create API Token'ボタンを押下する。
ダウンロード先は、
windowsなら
    C:\Users\<Windows-username>\.kaggle\kaggle.json
Macなら
    ~/.kaggle/kaggle.json
に指定。


ダウンロードした、kaggle.jsonのを開き、その内容を元に、

windows & bashなら
    C:\Users\<Windows-username>\.bashrc

mac & bashなら
    ~/.bashrc

に下記datadinosaurとxxxxを書き換えて追記。
```
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx
```



### コマンド

\<コンペティション名\>/\<ユーザー名\>/下で使用できるコマンド。
例えば/playground-series-s4e1/koichi/下で使用できる。

```
make download_data
```
使用するデータ(train.csv, test.csv, sumple_aubmission.csv)をダウンロードしてカレントディレクトリに配置

```
make submit
```
カレントディレクトリ内の"submission.csv"を提出。
