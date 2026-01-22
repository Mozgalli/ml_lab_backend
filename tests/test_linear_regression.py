import pytest
import numpy as np
from app.ml.algorithms import LinearRegressionFromScratch


def test_linear_regression_perfect_fit():
    """Mükemmel lineer ilişki testi"""
    # y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])
    
    model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y, method="gradient_descent")
    
    # R² skoru 1'e çok yakın olmalı
    r2 = model.score(X, y)
    assert r2 > 0.99
    
    # Katsayı ~2 olmalı
    assert abs(model.weights[0] - 2.0) < 0.1


def test_linear_regression_predict():
    """Tahmin testi"""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    model = LinearRegressionFromScratch()
    model.fit(X, y, method="normal_equation")
    
    # Yeni tahmin
    X_new = np.array([[4], [5]])
    predictions = model.predict(X_new)
    
    # Yaklaşık 8 ve 10 olmalı
    assert abs(predictions[0] - 8) < 1
    assert abs(predictions[1] - 10) < 1


def test_normal_equation_vs_gradient_descent():
    """İki yöntemin sonuçlarını karşılaştır"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    
    # Normal equation
    model1 = LinearRegressionFromScratch()
    model1.fit(X, y, method="normal_equation")
    
    # Gradient descent
    model2 = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=5000)
    model2.fit(X, y, method="gradient_descent")
    
    # İki yöntem de benzer sonuçlar vermeli
    assert abs(model1.weights[0] - model2.weights[0]) < 0.1
    assert abs(model1.bias - model2.bias) < 0.1
