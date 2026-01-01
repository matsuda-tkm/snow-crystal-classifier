"""OpenCV特徴量ベースの雪晶分類器（RandomForest）"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class SnowCrystalClassifier:
    """
    OpenCVを使用した画像処理特徴量に基づく雪晶分類器

    画像から特徴量を抽出し、RandomForestで霰(graupel)と雪片(snowflake)を分類する
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Args:
            n_estimators: 決定木の数
            max_depth: 決定木の最大深さ
            random_state: 乱数シード
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        """
        モデルを訓練する

        Args:
            X_train: 訓練画像データ (N, H, W, C)
            y_train: 訓練ラベル (N,)

        Returns:
            self
        """
        features = self._extract_features_batch(X_train)
        features = self.scaler.fit_transform(features)
        self.classifier.fit(features, y_train)
        return self

    def predict(self, X):
        """
        予測を行う

        Args:
            X: 入力画像データ (N, H, W, C)

        Returns:
            予測ラベル (N,)
        """
        features = self._extract_features_batch(X)
        features = self.scaler.transform(features)
        return self.classifier.predict(features)

    def _extract_features_batch(self, images):
        """複数画像から特徴量を抽出する"""
        return np.array([self._extract_features(img) for img in images])

    def _extract_features(self, image):
        """
        1枚の画像から特徴量を抽出する

        Args:
            image: RGB画像 (H, W, 3)

        Returns:
            特徴量ベクトル (1次元配列)
        """
        # カラー画像をグレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        features = []

        # ========================================
        # 形状特徴量（実装例）
        # ========================================
        # 二値化して輪郭を抽出
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            # 輪郭が見つからない場合はゼロで埋める
            features.extend([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        else:
            # 最大の輪郭を取得
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            # 円形度: 真円に近いほど1に近づく
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # 面積比: 画像全体に対する輪郭の面積
            area_ratio = area / (gray.shape[0] * gray.shape[1])

            # バウンディングボックス
            x, y, w, h = cv2.boundingRect(largest)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0

            # 凸包との比較
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # 複雑さ
            complexity = perimeter / np.sqrt(area) if area > 0 else 0

            # Huモーメント（形状の特徴を表す不変量）
            moments = cv2.moments(largest)
            hu = cv2.HuMoments(moments).flatten()[:4]
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

            features.extend([circularity, area_ratio, aspect_ratio, extent, solidity, complexity])
            features.extend(hu.tolist())

        # ========================================
        # テクスチャ特徴量（LBP + Gabor）
        # ========================================
        # LBP (Local Binary Pattern): 局所的なテクスチャパターンを捉える
        # 各ピクセルの周囲8近傍と比較し、パターンをエンコード
        padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for i in range(8):
            angle = i * np.pi / 4
            dy, dx = int(np.round(np.sin(angle))), int(np.round(np.cos(angle)))
            neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            lbp += ((neighbor >= gray).astype(np.uint8) << i)
        # LBPヒストグラム（16ビンに正規化）
        hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        features.extend((hist / (hist.sum() + 1e-7)).tolist())

        # Gaborフィルタ: 異なる方向・スケールのテクスチャを検出
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            for sigma in [3.0, 5.0]:
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10.0, 0.5, 0)
                filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
                features.extend([filtered.mean(), filtered.std()])

        # ========================================
        # エッジ特徴量
        # ========================================
        # Cannyエッジ検出: エッジの密度を計算
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.shape[0] * edges.shape[1] * 255)

        # Sobelフィルタ: 勾配の強度と方向
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        direction = np.arctan2(sobel_y, sobel_x)

        # 勾配方向のヒストグラム（8方向）
        dir_hist, _ = np.histogram(direction.ravel(), bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist / (dir_hist.sum() + 1e-7)

        # Laplacian: エッジの鮮明さ（分散が大きいほど鮮明）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000

        features.extend([edge_density, gradient.mean(), gradient.std(), laplacian_var])
        features.extend(dir_hist.tolist())

        # ========================================
        # 統計的特徴量
        # ========================================
        # 基本統計量
        mean, std = gray.mean(), gray.std()

        # エントロピー（情報量の尺度）
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-7)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        # 歪度と尖度（分布の形状）
        centered = gray.astype(np.float64) - mean
        skewness = np.mean(centered ** 3) / (std ** 3 + 1e-7)
        kurtosis = np.mean(centered ** 4) / (std ** 4 + 1e-7) - 3

        # パーセンタイル
        percentiles = np.percentile(gray.ravel(), [10, 25, 50, 75, 90]) / 255

        features.extend([
            mean / 255, std / 255, (gray.max() - gray.min()) / 255,
            entropy / 8, skewness, kurtosis
        ])
        features.extend(percentiles.tolist())

        return np.array(features)
