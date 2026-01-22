from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes

# FastAPI uygulaması oluştur
app = FastAPI(
    title="ML Lab Backend API",
    description="Makine Öğrenimi ve Matematik Temelleri için Backend API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API route'larını dahil et
app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
async def root():
    """Ana endpoint - API durumunu kontrol et"""
    return {
        "message": "ML Lab Backend API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü endpoint'i"""
    return {"status": "healthy"}
