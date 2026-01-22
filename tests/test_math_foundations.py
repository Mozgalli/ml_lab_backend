import pytest
import numpy as np
from app.math_foundations.core import LinearAlgebra, Statistics


def test_matrix_multiply():
    """Matris çarpımı testi"""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    result = LinearAlgebra.matrix_multiply(A, B)
    expected = np.array([[19, 22], [43, 50]])
    
    assert np.allclose(result, expected)


def test_transpose():
    """Matris transpozu testi"""
    A = np.array([[1, 2, 3], [4, 5, 6]])
    result = LinearAlgebra.transpose(A)
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    
    assert np.array_equal(result, expected)


def test_mean_calculation():
    """Ortalama hesaplama testi"""
    data = np.array([1, 2, 3, 4, 5])
    result = Statistics.mean(data)
    
    assert result == 3.0


def test_variance_calculation():
    """Varyans hesaplama testi"""
    data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    result = Statistics.variance(data)
    
    # Beklenen varyans: 4
    assert abs(result - 4.0) < 0.01


def test_normalization():
    """Z-score normalizasyon testi"""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    normalized = Statistics.normalize_zscore(data)
    
    # Normalize edilmiş verinin ortalaması ~0 olmalı
    assert abs(np.mean(normalized)) < 0.01
    
    # Standart sapma ~1 olmalı
    assert abs(np.std(normalized) - 1.0) < 0.1
