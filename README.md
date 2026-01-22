# 🧠 ML Lab Backend

**Makine Öğrenimi ve Matematik Temelleri** için kapsamlı bir Python Backend projesi. Docker ile containerize edilmiş, FastAPI ile REST API, Jupyter Notebook entegrasyonu ve sıfırdan yazılmış ML algoritmaları içerir.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Teknolojiler](#-teknolojiler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [API Endpoints](#-api-endpoints)
- [Proje Yapısı](#-proje-yapısı)
- [Öğrenme Yol Haritası](#-öğrenme-yol-haritası)
- [Algoritmalar](#-algoritmalar)

## ✨ Özellikler

### 🎯 Matematik Temelleri
- **Lineer Cebir**: Matris işlemleri, determinant, ters matris, eigenvalues/eigenvectors
- **İstatistik**: Ortalama, varyans, standart sapma, korelasyon, normalizasyon
- **Aktivasyon Fonksiyonları**: Sigmoid, ReLU, Tanh, Softmax
- **Kayıp Fonksiyonları**: MSE, MAE, Binary/Categorical Cross-Entropy
- **Optimizasyon**: Gradient Descent, Stochastic Gradient Descent

### 🤖 Makine Öğrenimi Algoritmaları
Tüm algoritmalar **sıfırdan Python ile yazılmıştır** (eğitim amaçlı):

1. **Linear Regression** - Sürekli değer tahmini
2. **Logistic Regression** - İkili sınıflandırma
3. **K-Means** - Kümeleme (clustering)
4. **K-Nearest Neighbors (KNN)** - Sınıflandırma ve regresyon
5. **Naive Bayes** - Olasılıksal sınıflandırma

### 🔧 Backend Özellikleri
- **FastAPI**: Modern, hızlı REST API
- **Docker**: Containerization ve kolay deployment
- **Jupyter Notebooks**: İnteraktif öğrenme ve deney
- **Automatic Documentation**: Swagger UI ve ReDoc
- **Type Hints**: Tip güvenliği

## 🛠 Teknolojiler

```
Backend:
├── Python 3.11
├── FastAPI
├── NumPy
├── Scikit-learn
├── Pandas
├── Matplotlib & Seaborn
└── Jupyter Notebook

Infrastructure:
├── Docker
├── Docker Compose
└── Uvicorn ASGI Server

Optional:
├── PyTorch
└── TensorFlow
```

## 📦 Kurulum

### Gereksinimler
- Docker & Docker Compose
- Git

### Adım 1: Projeyi Klonlayın
```bash
git clone <repository-url>
cd ml_lab_backend
```

### Adım 2: Environment Dosyasını Oluşturun
```bash
cp .env.example .env
```

### Adım 3: Docker ile Başlatın
```bash
# Tüm servisleri başlat
docker-compose up -d

# Logları takip et
docker-compose logs -f
```

### Adım 4: Servislere Erişin
- **Backend API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Jupyter Notebook**: http://localhost:8888

## 🚀 Kullanım

### 1. API ile İnteraktif Çalışma

**Swagger UI** üzerinden tüm endpoint'leri test edebilirsiniz:
```
http://localhost:8000/docs
```

### 2. Jupyter Notebook ile Öğrenme

```bash
# Jupyter container'ına bağlan
docker exec -it ml_lab_jupyter bash

# Veya tarayıcıdan direkt erişim
http://localhost:8888
```

### 3. Python Kodunda Kullanım

```python
import numpy as np
from app.ml.algorithms import LinearRegressionFromScratch
from app.math_foundations.core import Statistics

# Veri oluştur
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Model oluştur ve eğit
model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=1000)
model.fit(X, y, method="gradient_descent")

# Tahmin yap
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y)}")

# İstatistiksel analiz
stats = Statistics()
print(f"Mean: {stats.mean(y)}")
print(f"Std: {stats.standard_deviation(y)}")
```

## 🌐 API Endpoints

### Matematik İşlemleri

#### Vektör Normu
```bash
POST /api/v1/math/vector/norm
{
  "vector": [3, 4]
}
# Response: {"norm": 5.0, "explanation": "..."}
```

#### İç Çarpım
```bash
POST /api/v1/math/vector/dot-product
{
  "vector1": [1, 2, 3],
  "vector2": [4, 5, 6]
}
# Response: {"dot_product": 32.0, "explanation": "..."}
```

#### Matris Determinantı
```bash
POST /api/v1/math/matrix/determinant
{
  "matrix": [[1, 2], [3, 4]]
}
# Response: {"determinant": -2.0, "explanation": "..."}
```

#### Eigenvalues ve Eigenvectors
```bash
POST /api/v1/math/matrix/eigenvalues
{
  "matrix": [[4, 2], [1, 3]]
}
# Response: {"eigenvalues": [...], "eigenvectors": [...]}
```

### Makine Öğrenimi

#### Linear Regression Eğitimi
```bash
POST /api/v1/ml/linear-regression/train
{
  "X": [[1], [2], [3], [4], [5]],
  "y": [2, 4, 6, 8, 10]
}
# Response: {"coefficients": [...], "r2_score": 1.0, ...}
```

#### Gradient Descent Demo
```bash
POST /api/v1/ml/gradient-descent
{
  "X": [[1], [2], [3]],
  "y": [2, 4, 6]
}
# Response: {"final_parameters": [...], "cost_history": [...]}
```

#### Algoritma Listesi
```bash
GET /api/v1/ml/algorithms
# Response: Liste of all available algorithms
```

## 📁 Proje Yapısı

```
ml_lab_backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI uygulaması
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API endpoints
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   └── algorithms.py       # ML algoritmaları (scratch)
│   │
│   └── math_foundations/
│       ├── __init__.py
│       └── core.py             # Matematik fonksiyonları
│
├── notebooks/                   # Jupyter notebooks
├── data/                        # Dataset'ler
├── models/                      # Eğitilmiş modeller
├── tests/                       # Unit testler
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 🎓 Öğrenme Yol Haritası

### 1. Matematik Temelleri (Önce Buradan Başlayın!)

#### Lineer Cebir
- [ ] Vektör işlemleri (toplama, çarpma, norm)
- [ ] Matris çarpımı ve transpozu
- [ ] Determinant ve ters matris
- [ ] Eigenvalues ve eigenvectors
- [ ] SVD ve QR ayrıştırması

#### İstatistik
- [ ] Merkezi eğilim ölçüleri (mean, median, mode)
- [ ] Dağılım ölçüleri (variance, std)
- [ ] Kovaryans ve korelasyon
- [ ] Normalizasyon teknikleri

#### Kalkülüs
- [ ] Türev ve gradyan
- [ ] Kısmi türevler
- [ ] Chain rule
- [ ] Optimizasyon (gradient descent)

### 2. Makine Öğrenimi Algoritmaları

#### Supervised Learning
1. **Linear Regression** ⭐ Başlangıç
   - Normal equation
   - Gradient descent
   - Overfitting ve regularization
   
2. **Logistic Regression** ⭐⭐
   - Sigmoid fonksiyonu
   - Binary classification
   - Decision boundary
   
3. **K-Nearest Neighbors** ⭐⭐
   - Distance metrics
   - K değeri seçimi
   - Classification vs Regression

4. **Naive Bayes** ⭐⭐
   - Bayes teoremi
   - Probability distributions
   - Text classification

#### Unsupervised Learning
1. **K-Means Clustering** ⭐⭐
   - Centroid initialization
   - Convergence
   - Elbow method

### 3. İleri Seviye Konular
- Neural Networks
- Deep Learning
- Ensemble Methods
- Dimensionality Reduction (PCA)

## 🧪 Algoritmalar

### Linear Regression
```python
from app.ml.algorithms import LinearRegressionFromScratch

# Model oluştur
model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=1000)

# Eğit (iki yöntem)
model.fit(X_train, y_train, method="gradient_descent")
# veya
model.fit(X_train, y_train, method="normal_equation")

# Tahmin
predictions = model.predict(X_test)

# Değerlendirme
r2 = model.score(X_test, y_test)
```

### Logistic Regression
```python
from app.ml.algorithms import LogisticRegressionFromScratch

model = LogisticRegressionFromScratch(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Olasılık tahmini
probabilities = model.predict_proba(X_test)

# Sınıf tahmini
predictions = model.predict(X_test, threshold=0.5)

# Accuracy
accuracy = model.score(X_test, y_test)
```

### K-Means
```python
from app.ml.algorithms import KMeansFromScratch

model = KMeansFromScratch(n_clusters=3, max_iters=100, random_state=42)
model.fit(X)

# Küme etiketleri
labels = model.labels

# Centroid'ler
centroids = model.centroids

# Yeni veri için tahmin
new_labels = model.predict(X_new)
```

### K-Nearest Neighbors
```python
from app.ml.algorithms import KNNFromScratch

model = KNNFromScratch(k=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

### Naive Bayes
```python
from app.ml.algorithms import NaiveBayesFromScratch

model = NaiveBayesFromScratch()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

## 📊 Örnek Kullanım Senaryoları

### 1. Ev Fiyat Tahmini (Linear Regression)
```python
# Veri: [alan, oda_sayisi, yaş] -> fiyat
X = np.array([
    [100, 3, 10],
    [150, 4, 5],
    [80, 2, 15],
    [200, 5, 2]
])
y = np.array([250000, 400000, 180000, 550000])

model = LinearRegressionFromScratch()
model.fit(X, y)

# Yeni ev için tahmin
new_house = np.array([[120, 3, 8]])
predicted_price = model.predict(new_house)
```

### 2. Email Spam Tespiti (Logistic Regression)
```python
# Feature'lar (kelime frekansları, büyük harf oranı, vs.)
X_emails = ...  # Email feature'ları
y_spam = ...    # 0: normal, 1: spam

model = LogisticRegressionFromScratch()
model.fit(X_emails, y_spam)

# Yeni email test et
new_email_features = ...
spam_probability = model.predict_proba(new_email_features)
```

### 3. Müşteri Segmentasyonu (K-Means)
```python
# Müşteri özellikleri: [yaş, gelir, harcama]
X_customers = np.array([...])

model = KMeansFromScratch(n_clusters=4)
model.fit(X_customers)

# Her müşterinin segmenti
segments = model.labels
```

## 🐳 Docker Komutları

```bash
# Servisleri başlat
docker-compose up -d

# Logları görüntüle
docker-compose logs -f ml_backend

# Backend container'a bağlan
docker exec -it ml_lab_backend bash

# Jupyter container'a bağlan
docker exec -it ml_lab_jupyter bash

# Servisleri durdur
docker-compose down

# Servisleri yeniden build et
docker-compose up --build -d

# Tüm verileri sil (dikkat!)
docker-compose down -v
```

## 🧪 Test

```bash
# Docker içinde test çalıştır
docker exec -it ml_lab_backend pytest

# Veya local
pytest tests/
```

## 📚 Kaynaklar ve Öğrenme Materyalleri

### Kitaplar
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Deep Learning" - Ian Goodfellow

### Online Kurslar
- Andrew Ng - Machine Learning (Coursera)
- Fast.ai - Practical Deep Learning
- MIT 18.06 - Linear Algebra

### Matematiksel Formüller

#### Gradient Descent
$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

#### Linear Regression (Normal Equation)
$$\theta = (X^TX)^{-1}X^Ty$$

#### Sigmoid Function
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

#### Mean Squared Error
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

## 🤝 Katkıda Bulunma

Projeye katkıda bulunmak isterseniz:
1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje eğitim amaçlıdır ve açık kaynaklıdır.

## 💡 Sorular ve Destek

Sorularınız için:
- Issue açın
- Discussions kullanın
- Email: [your-email]

## 🎯 Roadmap

- [ ] Daha fazla algoritma ekle (Decision Trees, Random Forest)
- [ ] Neural Network implementasyonu
- [ ] Web UI ekle (React frontend)
- [ ] Model persistence (model kaydetme/yükleme)
- [ ] Daha fazla örnek dataset
- [ ] Video tutorial serisi
- [ ] Interactive visualizations

---

**Happy Learning! 🚀🧠**

*Makine öğrenimi yolculuğunuzda başarılar dileriz!*
