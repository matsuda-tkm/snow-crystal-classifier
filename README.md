# 雪晶分類器 (Snow Crystal Classifier)

雪の結晶画像から「霰（あられ）」と「雪片」を自動で見分ける機械学習プログラムです。


## 課題について

このリポジトリは学習用のテンプレートです。以下の2つの課題を実装してください。

### 課題1: クロスバリデーションの実装

`main.py`の`run_cross_validation`関数を実装してください。

実装すべき処理:
1. StratifiedKFoldを使ってデータをn_folds個に分割する
2. 各Foldで訓練と評価を繰り返す
3. 全Foldの結果を集計して返す

参考:
- scikit-learn StratifiedKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

<details>
<summary>💡 解答例と解説（クリックで展開）</summary>

#### 実装のポイント

```python
def run_cross_validation(X, y, n_folds=5, random_seed=42):
    # 1. StratifiedKFoldでデータを分割
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_metrics = []  # 各Foldの評価結果
    all_preds = []     # 全予測結果（混同行列用）
    all_labels = []
    
    # 2. 各Foldで訓練と評価
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = SnowCrystalClassifier(random_state=random_seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = compute_metrics(y_test, y_pred)
        fold_metrics.append(metrics)
        all_preds.extend(y_pred)
        all_labels.extend(y_test)
    
    # 3. 平均と標準偏差を計算
    mean_metrics = {
        "accuracy": np.mean([m["accuracy"] for m in fold_metrics]),
        # ... 他の指標も同様
    }
    std_metrics = {
        "accuracy": np.std([m["accuracy"] for m in fold_metrics]),
        # ...
    }
    return mean_metrics, std_metrics
```

#### 解説

1. **StratifiedKFold**: `shuffle=True`でデータをシャッフルし、`random_state`で再現性を確保します。`split(X, y)`は訓練用とテスト用のインデックスを返します。

2. **各Foldでの処理**: インデックスを使って`X[train_idx]`のようにデータを分割し、分類器の訓練→予測→評価を行います。

3. **結果の集計**: `np.mean()`と`np.std()`で平均と標準偏差を計算します。混同行列は全Foldの予測結果をまとめて計算します。

</details>

### 課題2: 特徴量抽出の実装

`classifier.py`の`_extract_features`メソッドに追加の特徴量を実装してください。

現在は形状特徴量（10次元）のみ実装されています。OpenCVのドキュメントを参照して、分類に有効な特徴量を追加してください。

参考:
- OpenCV公式ドキュメント: https://docs.opencv.org/4.x/
- 画像処理チュートリアル: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html

<details>
<summary>💡 解答例と解説（クリックで展開）</summary>

#### 追加する特徴量

| カテゴリ | 特徴量 | 次元数 | 説明 |
|---------|--------|--------|------|
| テクスチャ | LBPヒストグラム | 16 | 局所的なテクスチャパターン |
| テクスチャ | Gaborフィルタ | 16 | 方向・スケール別のテクスチャ |
| エッジ | Canny密度 | 1 | エッジの密度 |
| エッジ | 勾配統計 | 3 | 勾配の平均・標準偏差・Laplacian分散 |
| エッジ | 方向ヒストグラム | 8 | 勾配方向の分布 |
| 統計 | 基本統計量 | 6 | 平均、標準偏差、レンジ、エントロピー、歪度、尖度 |
| 統計 | パーセンタイル | 5 | 10%, 25%, 50%, 75%, 90% |

#### 実装例

```python
# テクスチャ特徴量（LBP）
# 各ピクセルの周囲8近傍と比較してパターンをエンコード
padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
h, w = gray.shape
lbp = np.zeros((h, w), dtype=np.uint8)
for i in range(8):
    angle = i * np.pi / 4
    dy, dx = int(np.round(np.sin(angle))), int(np.round(np.cos(angle)))
    neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
    lbp += ((neighbor >= gray).astype(np.uint8) << i)
hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
features.extend((hist / (hist.sum() + 1e-7)).tolist())

# Gaborフィルタ（4方向 × 2スケール）
for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
    for sigma in [3.0, 5.0]:
        kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0)
        filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
        features.extend([filtered.mean(), filtered.std()])

# エッジ特徴量
edges = cv2.Canny(gray, 50, 150)
edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
direction = np.arctan2(sobel_y, sobel_x)

dir_hist, _ = np.histogram(direction.ravel(), bins=8, range=(-np.pi, np.pi))
dir_hist = dir_hist / (dir_hist.sum() + 1e-7)

laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
features.extend([edge_density, gradient.mean(), gradient.std(), laplacian_var])
features.extend(dir_hist.tolist())

# 統計的特徴量
mean, std = gray.mean(), gray.std()
hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
hist = hist / (hist.sum() + 1e-7)
entropy = -np.sum(hist * np.log2(hist + 1e-7))

centered = gray.astype(np.float64) - mean
skewness = np.mean(centered ** 3) / (std ** 3 + 1e-7)
kurtosis = np.mean(centered ** 4) / (std ** 4 + 1e-7) - 3

percentiles = np.percentile(gray.ravel(), [10, 25, 50, 75, 90]) / 255

features.extend([mean / 255, std / 255, (gray.max() - gray.min()) / 255,
                 entropy / 8, skewness, kurtosis])
features.extend(percentiles.tolist())
```

#### 解説

1. **LBP (Local Binary Pattern)**: 各ピクセルの周囲8近傍と中心ピクセルを比較し、大きければ1、小さければ0としてビットパターンを作成します。テクスチャの局所的なパターンを捉えるのに有効です。

2. **Gaborフィルタ**: 特定の方向と周波数に反応するフィルタです。異なる方向（0°, 45°, 90°, 135°）とスケール（σ=3, 5）で適用し、雪片の複雑な模様を検出します。

3. **Cannyエッジ検出**: エッジを検出し、その密度を計算します。雪片はエッジが多く、霰はエッジが少ない傾向があります。

4. **Sobelフィルタ**: x方向とy方向の勾配を計算し、勾配の強度と方向を求めます。方向のヒストグラムは、雪片の六角形パターンを捉えるのに役立ちます。

5. **統計的特徴量**: 画像の明るさの分布を表す指標です。エントロピーは情報量、歪度・尖度は分布の形状を表します。

</details>


## このプロジェクトについて

### 何をするプログラムか

雪の結晶には様々な形があります。このプログラムは、雪の結晶の画像を見て、それが「霰 (graupel)」なのか「雪片 (snowflake)」なのかを判定します。

- *graupel*（霰）: 丸っこい粒状の雪。雲の中で雪の結晶に水滴がくっついてできる
- *snowflake*（雪片）: 六角形の美しい形をした雪の結晶

### どうやって見分けるのか

人間が雪の結晶を見分けるとき、「形が丸いか」「複雑な模様があるか」などの特徴を見ています。このプログラムも同じように、画像から特徴を数値として取り出し、その数値をもとに判定します。

| 特徴の種類 | 何を見ているか | 例 |
|-----------|--------------|-----|
| 形状特徴 | 輪郭の形、丸さ、複雑さ | 霰は丸い、雪片は複雑 |
| テクスチャ特徴 | 表面の模様、パターン | 雪片は細かい模様がある |
| エッジ特徴 | 輪郭の鮮明さ、方向 | 雪片はエッジが多い |
| 統計特徴 | 明るさの分布 | 全体的な明暗の傾向 |

### 機械学習の仕組み

今回は *RandomForest*（ランダムフォレスト）という手法を使ってみましょう。

RandomForestは「たくさんの決定木を作って、多数決で答えを決める」方法です。

決定木とは、「もし〇〇なら→次の質問へ」という分岐を繰り返して答えにたどり着く仕組みです。例えば以下のような判断をします。

```
丸さが0.7以上？
├── はい → 霰
└── いいえ → エッジの数が多い？
              ├── はい → 雪片
              └── いいえ → 霰
```

1本の決定木だと間違えやすいですが、100本の決定木を作って多数決を取ることで、より正確な判定ができます。これがRandomForestの考え方です。

### 評価方法

機械学習モデルの精度を測るために *クロスバリデーション*（交差検証）という方法を使っています。

データを5つのグループに分け、「4グループで学習→1グループでテスト」を5回繰り返します。こうすることで、すべてのデータがテストに使われ、偏りのない評価ができます。

```
1回目: [テスト][学習][学習][学習][学習]
2回目: [学習][テスト][学習][学習][学習]
3回目: [学習][学習][テスト][学習][学習]
4回目: [学習][学習][学習][テスト][学習]
5回目: [学習][学習][学習][学習][テスト]
```

今回は *StratifiedKFold*（層別K分割）を使用してみましょう。通常のランダム分割では、あるグループに霰ばかり、別のグループに雪片ばかりが偏る可能性があります。層別分割では、各グループにおけるクラスの比率が元のデータと同じになるように分割します。

```
元データ: 霰25% / 雪片75%
  ↓ 層別分割
各グループ: 霰25% / 雪片75%（比率を維持）
```

### 評価指標の読み方

結果に表示される指標の意味は以下の通りです。

| 指標 | 意味 | この値が高いと |
|-----|------|--------------|
| *Accuracy* | 全体の正解率 | 全体的に正しく判定できている |
| *Precision* | 「霰」と判定したもののうち、本当に霰だった割合 | 誤検出が少ない |
| *Recall* | 本当に霰のもののうち、正しく「霰」と判定できた割合 | 見逃しが少ない |
| *F1* | PrecisionとRecallのバランスを取った指標 | 総合的に良い性能 |

## 使い方

### 必要な環境

- uv（パッケージマネージャ）

### インストール

```bash
git clone https://github.com/matsuda-tkm/snow-crystal-classifier.git
cd snow-crystal-classifier
uv sync
```

### 分類器の実行

```bash
uv run main.py
```

実行すると、5分割クロスバリデーションで評価を行い、結果を `output/` フォルダに保存します。

オプション:

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| --data-dir | データセットのパス | dataset |
| --output-dir | 出力先のパス | output |
| --n-folds | クロスバリデーションの分割数 | 5 |
| --image-size | 画像のリサイズサイズ | 128 |
| --seed | 乱数シード | 42 |

### 決定木の可視化

決定木がどのような判断をしているか可視化できます。

```bash
uv run visualize_tree.py
```

実行すると、以下のファイルが生成されます:

| ファイル | 内容 |
|---------|------|
| decision_tree.png | 決定木の構造図（どの特徴量でどう分岐しているか） |
| feature_importance.png | 特徴量の重要度ランキング |
| decision_tree_rules.txt | 決定木のルールをテキストで出力 |

RandomForestは100本の決定木の多数決ですが、このスクリプトでは1本の決定木を訓練して可視化します。どの特徴量が分類に効いているかを確認するのに役立ちます。

## データセット

`dataset/` フォルダに以下の構成で画像を配置してください。

```
dataset/
├── graupel/      # 霰の画像 (107枚)
│   ├── image1.png
│   └── ...
└── snowflake/    # 雪片の画像 (323枚)
    ├── image1.png
    └── ...
```

## 出力ファイル

実行後、`output/` フォルダに以下のファイルが生成されます。

| ファイル | 内容 |
|---------|------|
| confusion_matrix.png | 混同行列（予測と正解の対応表） |
| metrics.png | 評価指標のグラフ |
| results.csv | 評価指標の数値データ |

## ファイル構成

```
snow-crystal-classifier/
├── main.py              # メインスクリプト（クロスバリデーション評価）
├── classifier.py        # 分類器の実装
├── visualize_tree.py    # 決定木の可視化
├── pyproject.toml       # 依存関係の定義
├── dataset/             # データセット
└── output/              # 出力結果
```

## 使用ライブラリ

- OpenCV: 画像処理と特徴量抽出
- scikit-learn: RandomForest分類器と評価
- NumPy: 数値計算
- Matplotlib / Seaborn: 可視化
