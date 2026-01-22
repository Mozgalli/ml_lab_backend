"""
Matematik Temelleri Modülü
Lineer cebir, istatistik ve optimizasyon fonksiyonları
"""

import numpy as np
from typing import List, Tuple, Union


class LinearAlgebra:
    """Lineer Cebir işlemleri"""
    
    @staticmethod
    def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Matris çarpımı: C = A × B
        
        Args:
            A: m×n matris
            B: n×p matris
        Returns:
            C: m×p matris
        """
        return np.dot(A, B)
    
    @staticmethod
    def transpose(matrix: np.ndarray) -> np.ndarray:
        """Matris transpozu: Aᵀ"""
        return matrix.T
    
    @staticmethod
    def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Gram-Schmidt Ortonormalizasyon
        Vektör setini ortogonal ve normalize edilmiş hale getirir
        """
        orthonormal = []
        for v in vectors:
            # Önceki vektörlerden projeksiyon çıkar
            w = v.copy()
            for u in orthonormal:
                w = w - np.dot(v, u) * u
            # Normalize et
            if np.linalg.norm(w) > 1e-10:
                w = w / np.linalg.norm(w)
                orthonormal.append(w)
        return orthonormal
    
    @staticmethod
    def svd_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Singular Value Decomposition (SVD)
        A = U Σ Vᵀ
        """
        U, s, Vt = np.linalg.svd(matrix)
        return U, s, Vt
    
    @staticmethod
    def qr_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR Ayrıştırması
        A = QR (Q: ortogonal, R: üst üçgen)
        """
        return np.linalg.qr(matrix)


class Statistics:
    """İstatistik işlemleri"""
    
    @staticmethod
    def mean(data: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
        """Aritmetik ortalama: μ = (1/n)Σxᵢ"""
        return np.mean(data, axis=axis)
    
    @staticmethod
    def variance(data: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
        """Varyans: σ² = (1/n)Σ(xᵢ - μ)²"""
        return np.var(data, axis=axis)
    
    @staticmethod
    def standard_deviation(data: np.ndarray, axis: int = None) -> Union[float, np.ndarray]:
        """Standart sapma: σ = √(variance)"""
        return np.std(data, axis=axis)
    
    @staticmethod
    def covariance_matrix(data: np.ndarray) -> np.ndarray:
        """
        Kovaryans matrisi
        Her iki değişken arasındaki ilişkiyi ölçer
        """
        return np.cov(data.T)
    
    @staticmethod
    def correlation_matrix(data: np.ndarray) -> np.ndarray:
        """
        Korelasyon matrisi
        Normalize edilmiş kovaryans (-1 ile 1 arası)
        """
        return np.corrcoef(data.T)
    
    @staticmethod
    def normalize_zscore(data: np.ndarray) -> np.ndarray:
        """
        Z-score normalizasyonu
        z = (x - μ) / σ
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-10)
    
    @staticmethod
    def normalize_minmax(data: np.ndarray) -> np.ndarray:
        """
        Min-Max normalizasyonu
        x' = (x - min) / (max - min)
        """
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-10)


class ActivationFunctions:
    """Aktivasyon fonksiyonları (Sinir ağları için)"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Sigmoid: σ(x) = 1 / (1 + e⁻ˣ)
        Çıktı aralığı: (0, 1)
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Sigmoid türevi: σ'(x) = σ(x)(1 - σ(x))"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        ReLU: f(x) = max(0, x)
        En popüler aktivasyon fonksiyonu
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU türevi: f'(x) = 1 if x > 0 else 0"""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """
        Tanh: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
        Çıktı aralığı: (-1, 1)
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Tanh türevi: tanh'(x) = 1 - tanh²(x)"""
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax: σ(xᵢ) = eˣⁱ / Σeˣʲ
        Çok sınıflı sınıflandırma için (olasılık dağılımı)
        """
        exp_x = np.exp(x - np.max(x))  # Numerik stabilite için
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class LossFunctions:
    """Kayıp fonksiyonları"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error (MSE)
        L = (1/n)Σ(yᵢ - ŷᵢ)²
        Regresyon problemleri için
        """
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE)
        L = (1/n)Σ|yᵢ - ŷᵢ|
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy
        L = -Σ[y log(ŷ) + (1-y)log(1-ŷ)]
        İkili sınıflandırma için
        """
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Categorical Cross-Entropy
        Çok sınıflı sınıflandırma için
        """
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


class Optimization:
    """Optimizasyon algoritmaları"""
    
    @staticmethod
    def gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        n_iterations: int = 1000
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Vanilla Gradient Descent
        θ = θ - α∇J(θ)
        """
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        
        for _ in range(n_iterations):
            predictions = X.dot(theta)
            errors = predictions - y
            gradient = (1/m) * X.T.dot(errors)
            theta = theta - learning_rate * gradient
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(cost)
        
        return theta, cost_history
    
    @staticmethod
    def stochastic_gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        batch_size: int = 1
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Stochastic Gradient Descent (SGD)
        Her adımda rastgele bir örnek kullanır
        """
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        
        for iteration in range(n_iterations):
            indices = np.random.randint(0, m, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            
            predictions = X_batch.dot(theta)
            errors = predictions - y_batch
            gradient = (1/batch_size) * X_batch.T.dot(errors)
            theta = theta - learning_rate * gradient
            
            # Tüm veri için maliyet hesapla
            if iteration % 10 == 0:
                all_predictions = X.dot(theta)
                all_errors = all_predictions - y
                cost = (1/(2*m)) * np.sum(all_errors**2)
                cost_history.append(cost)
        
        return theta, cost_history
