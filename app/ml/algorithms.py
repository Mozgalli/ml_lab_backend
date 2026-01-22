"""
ML Algorithms Module
Sıfırdan yazılmış makine öğrenimi algoritmaları (eğitim amaçlı)
"""

import numpy as np
from typing import Optional


class LinearRegressionFromScratch:
    """
    Sıfırdan Lineer Regresyon
    
    Model: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    Optimizasyon: Gradient Descent veya Normal Equation
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "gradient_descent"):
        """
        Modeli eğit
        
        Args:
            X: Feature matrisi (n_samples, n_features)
            y: Hedef değerler (n_samples,)
            method: 'gradient_descent' veya 'normal_equation'
        """
        n_samples, n_features = X.shape
        
        if method == "normal_equation":
            # Normal Equation: θ = (XᵀX)⁻¹Xᵀy
            X_b = np.c_[np.ones((n_samples, 1)), X]
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = theta[0]
            self.weights = theta[1:]
        
        else:  # gradient_descent
            self.weights = np.zeros(n_features)
            self.bias = 0
            
            for i in range(self.n_iterations):
                # Tahmin: ŷ = Xθ + b
                y_pred = np.dot(X, self.weights) + self.bias
                
                # Gradyan hesapla
                dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
                db = (1/n_samples) * np.sum(y_pred - y)
                
                # Parametreleri güncelle
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Maliyet (MSE)
                cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
                self.cost_history.append(cost)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahmin yap"""
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score hesapla"""
        y_pred = self.predict(X)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - (ss_res / ss_tot)


class LogisticRegressionFromScratch:
    """
    Sıfırdan Lojistik Regresyon (İkili Sınıflandırma)
    
    Model: P(y=1|x) = σ(θᵀx) = 1 / (1 + e⁻ᶿᵀˣ)
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid aktivasyon fonksiyonu"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Modeli eğit"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            
            # Sigmoid aktivasyon
            y_pred = self.sigmoid(linear_model)
            
            # Gradyan hesapla
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Parametreleri güncelle
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Binary cross-entropy loss
            epsilon = 1e-10
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            cost = -(1/n_samples) * np.sum(
                y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped)
            )
            self.cost_history.append(cost)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Olasılık tahminleri"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Sınıf tahminleri (0 veya 1)"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy hesapla"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class KMeansFromScratch:
    """
    Sıfırdan K-Means Clustering
    
    Gözetimsiz öğrenme: Veriyi K kümeye ayırır
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def fit(self, X: np.ndarray):
        """K-Means algoritmasını uygula"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        # Rastgele başlangıç centroidleri seç
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            # Her noktayı en yakın centroide ata
            self.labels = self._assign_clusters(X)
            
            # Yeni centroidleri hesapla
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0) 
                for k in range(self.n_clusters)
            ])
            
            # Convergence kontrolü
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return self
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Her noktayı en yakın centroide ata"""
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self.centroids
        ])
        return np.argmin(distances, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Yeni noktalar için küme tahminleri"""
        return self._assign_clusters(X)
    
    def inertia(self, X: np.ndarray) -> float:
        """Within-cluster sum of squares"""
        return sum(
            np.sum((X[self.labels == k] - self.centroids[k])**2)
            for k in range(self.n_clusters)
        )


class KNNFromScratch:
    """
    Sıfırdan K-Nearest Neighbors (KNN)
    
    Hem sınıflandırma hem de regresyon için kullanılabilir
    """
    
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Eğitim verilerini sakla (KNN lazy learning algoritmasıdır)"""
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Euclidean mesafe hesapla"""
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahminler yap"""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray):
        """Tek bir nokta için tahmin"""
        # Tüm eğitim noktalarına olan mesafeleri hesapla
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # En yakın k komşuyu bul
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # En çok tekrar eden sınıfı seç (classification)
        # veya ortalamayı al (regression)
        if isinstance(k_nearest_labels[0], (int, np.integer)):
            # Classification
            return np.bincount(k_nearest_labels).argmax()
        else:
            # Regression
            return np.mean(k_nearest_labels)


class NaiveBayesFromScratch:
    """
    Sıfırdan Naive Bayes Classifier
    
    Bayes Teoremi: P(y|X) = P(X|y) * P(y) / P(X)
    """
    
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Modeli eğit"""
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Her sınıf için istatistikleri hesapla
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples
        
        return self
    
    def _gaussian_pdf(self, class_idx: int, x: np.ndarray) -> float:
        """Gaussian olasılık yoğunluk fonksiyonu"""
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return np.prod(numerator / denominator)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Tahminler yap"""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray):
        """Tek bir örnek için tahmin"""
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            likelihood = np.sum(np.log(self._gaussian_pdf(idx, x) + 1e-10))
            posterior = prior + likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
