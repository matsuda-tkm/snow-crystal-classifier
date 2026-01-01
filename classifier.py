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
        # TODO: 追加の特徴量を実装してください
        # ========================================
        # OpenCVのドキュメントを参照し、画像から特徴量を抽出してください
        # 参考: https://docs.opencv.org/4.x/
        #
        # 例:
        # - エッジ検出 (Canny, Sobel, Laplacian)
        # - フィルタ処理 (blur, GaussianBlur, filter2D)
        # - ヒストグラム (calcHist)
        # - その他の画像処理関数
        #
        # 抽出した特徴量は features.extend([...]) で追加してください

        return np.array(features)
