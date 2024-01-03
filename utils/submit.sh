COUNT_FILE="count.txt"
SUBMIT_PROJECT=$1

# ファイルが存在しなければ0からスタート
if [ ! -f "$COUNT_FILE" ]; then
    echo 0 > "$COUNT_FILE"
fi

# カウントを読み取り、インクリメントして更新
COUNT=$(<"$COUNT_FILE")
echo $((COUNT + 1)) > "$COUNT_FILE"
kaggle competitions submit -c $SUBMIT_PROJECT -f submission.csv -m "submission $(<"$COUNT_FILE")"

# カウントの値を表示
echo "Iteration: $(<"$COUNT_FILE")"
