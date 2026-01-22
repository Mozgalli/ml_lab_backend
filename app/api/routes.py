from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

router = APIRouter()

# Pydantic modelleri
class VectorInput(BaseModel):
    vector: List[float]

class MatrixInput(BaseModel):
    matrix: List[List[float]]

class TwoVectorsInput(BaseModel):
    vector1: List[float]
    vector2: List[float]

class LinearRegressionInput(BaseModel):
    X: List[List[float]]
    y: List[float]

class PredictionInput(BaseModel):
    X: List[List[float]]


# ==================== MATEMATİK TEMELLERİ ====================

@router.post("/math/vector/norm")
async def calculate_vector_norm(data: VectorInput):
    """Vektör normunu hesapla (L2 norm)"""
    try:
        vec = np.array(data.vector)
        norm = np.linalg.norm(vec)
        return {
            "vector": data.vector,
            "norm": float(norm),
            "explanation": "L2 norm (Euclidean norm) hesaplandı: √(x₁² + x₂² + ... + xₙ²)"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/math/vector/dot-product")
async def calculate_dot_product(data: TwoVectorsInput):
    """İki vektörün iç çarpımını hesapla"""
    try:
        v1 = np.array(data.vector1)
        v2 = np.array(data.vector2)
        
        if len(v1) != len(v2):
            raise ValueError("Vektörler aynı boyutta olmalı")
        
        dot_product = np.dot(v1, v2)
        return {
            "vector1": data.vector1,
            "vector2": data.vector2,
            "dot_product": float(dot_product),
            "explanation": "İç çarpım: v₁·v₂ = Σ(v₁ᵢ * v₂ᵢ)"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/math/matrix/determinant")
async def calculate_determinant(data: MatrixInput):
    """Matris determinantını hesapla"""
    try:
        matrix = np.array(data.matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matris kare matris olmalı")
        
        det = np.linalg.det(matrix)
        return {
            "matrix": data.matrix,
            "determinant": float(det),
            "explanation": "Determinant, matrisin 'hacmini' ölçer"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/math/matrix/inverse")
async def calculate_inverse(data: MatrixInput):
    """Matris tersini hesapla"""
    try:
        matrix = np.array(data.matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matris kare matris olmalı")
        
        inverse = np.linalg.inv(matrix)
        return {
            "matrix": data.matrix,
            "inverse": inverse.tolist(),
            "explanation": "A⁻¹ hesaplandı. A * A⁻¹ = I (birim matris)"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/math/matrix/eigenvalues")
async def calculate_eigenvalues(data: MatrixInput):
    """Öz değerleri ve öz vektörleri hesapla"""
    try:
        matrix = np.array(data.matrix)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matris kare matris olmalı")
        
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return {
            "matrix": data.matrix,
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": eigenvectors.tolist(),
            "explanation": "Av = λv denklemini sağlayan λ (eigenvalue) ve v (eigenvector) değerleri"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== MAKİNE ÖĞRENİMİ ALGORİTMALARI ====================

@router.post("/ml/linear-regression/train")
async def train_linear_regression(data: LinearRegressionInput):
    """
    Lineer Regresyon modeli eğit
    y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        
        X = np.array(data.X)
        y = np.array(data.y)
        
        # Model oluştur ve eğit
        model = LinearRegression()
        model.fit(X, y)
        
        # Tahminler ve metrikler
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "mse": float(mse),
            "r2_score": float(r2),
            "predictions": y_pred.tolist(),
            "explanation": "Model başarıyla eğitildi. R² skoru modelin açıklama gücünü gösterir (1'e yakın = daha iyi)"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ml/gradient-descent")
async def gradient_descent_demo(data: LinearRegressionInput):
    """
    Gradient Descent algoritması ile lineer regresyon
    Elle yazılmış implementasyon (eğitim amaçlı)
    """
    try:
        X = np.array(data.X)
        y = np.array(data.y).reshape(-1, 1)
        
        # Bias terimi ekle
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Parametreleri başlat
        theta = np.random.randn(X_b.shape[1], 1)
        learning_rate = 0.01
        n_iterations = 1000
        m = len(y)
        
        cost_history = []
        
        # Gradient Descent
        for iteration in range(n_iterations):
            # Tahmin
            predictions = X_b.dot(theta)
            
            # Hata (loss)
            errors = predictions - y
            
            # Maliyet fonksiyonu (MSE)
            cost = (1/(2*m)) * np.sum(errors**2)
            cost_history.append(float(cost))
            
            # Gradyan hesapla
            gradients = (1/m) * X_b.T.dot(errors)
            
            # Parametreleri güncelle
            theta = theta - learning_rate * gradients
        
        return {
            "final_parameters": theta.flatten().tolist(),
            "initial_cost": cost_history[0],
            "final_cost": cost_history[-1],
            "cost_history": cost_history[::100],  # Her 100. iterasyon
            "iterations": n_iterations,
            "learning_rate": learning_rate,
            "explanation": "Gradient Descent: θ = θ - α∇J(θ) formülü ile parametreler optimize edildi"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ml/algorithms")
async def list_algorithms():
    """Mevcut ML algoritmalarını listele"""
    return {
        "algorithms": [
            {
                "name": "Linear Regression",
                "type": "Supervised - Regression",
                "description": "Sürekli değerler tahmin eder",
                "complexity": "O(n²) - O(n³)"
            },
            {
                "name": "Logistic Regression", 
                "type": "Supervised - Classification",
                "description": "İkili sınıflandırma için kullanılır",
                "complexity": "O(n²)"
            },
            {
                "name": "K-Means",
                "type": "Unsupervised - Clustering",
                "description": "Veriyi k kümeye ayırır",
                "complexity": "O(n*k*i)"
            },
            {
                "name": "Decision Trees",
                "type": "Supervised - Classification/Regression",
                "description": "Karar ağacı yapısı oluşturur",
                "complexity": "O(n*log(n)*d)"
            }
        ]
    }
