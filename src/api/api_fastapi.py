# Instalar: pip install fastapi uvicorn scikit-learn pandas mlflow numpy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import mlflow
import numpy as np
import pandas as pd
import os
from typing import List, Dict

# --- Configuración y Carga del Modelo ---
MODEL_NAME = "ModeloRecomendacion_v1"
model_knn = None
item_cols = None
user_cols = None
R_df = None

def load_model_and_data():
    """Cargar modelo y datos necesarios para las recomendaciones"""
    global model_knn, item_cols, user_cols, R_df
    
    try:
        # Opción 1: Cargar desde MLflow Registry (modelo registrado)
        model_uri = f"models:/{MODEL_NAME}/1"  # Versión 1 del modelo registrado
        import mlflow.sklearn as mlflow_sklearn
        model_knn = mlflow_sklearn.load_model(model_uri)
        print("✅ Modelo cargado desde MLflow Registry")
        
        # Cargar mapeos desde archivos locales (más simple y confiable)
        item_cols = np.load('data/item_cols.npy', allow_pickle=True).tolist()
        user_cols = np.load('data/user_cols.npy', allow_pickle=True).tolist()
        print("✅ Mapeos cargados desde archivos locales")
            
    except Exception as e:
        print(f"⚠️ Error cargando desde MLflow: {e}")
        try:
            # Fallback: cargar archivos locales y recrear modelo
            item_cols = np.load('data/item_cols.npy', allow_pickle=True).tolist()
            user_cols = np.load('data/user_cols.npy', allow_pickle=True).tolist()
            
            # Cargar datos para reconstruir la matriz
            df = pd.read_csv('data/ratings.csv', 
                           names=['user_id', 'item_id', 'rating', 'timestamp'],
                           sep='\t')
            
            # Filtrar datos igual que en el entrenamiento
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()
            active_users = user_counts[user_counts >= 5].index
            popular_items = item_counts[item_counts >= 5].index
            df_filtered = df[df['user_id'].isin(active_users) & df['item_id'].isin(popular_items)]
            
            # Recrear matriz y modelo
            R_df = df_filtered.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
            
            from sklearn.neighbors import NearestNeighbors
            model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
            model_knn.fit(R_df.values)
            
            print("✅ Modelo recreado desde datos locales")
            
        except Exception as e2:
            print(f"❌ Error crítico: {e2}")
            raise Exception("No se pudo cargar el modelo ni los datos")

# Cargar modelo al iniciar la aplicación
load_model_and_data()


app = FastAPI(title="API de Recomendaciones", description="Sistema de recomendación de películas con kNN")

# --- Modelos de datos para la API ---
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict]
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    items_count: int
    users_count: int

# --- Función de Recomendación (Lógica de Negocio) ---
def get_recommendations_for_user(user_id: int, n_recommendations: int = 5):
    """Obtener recomendaciones reales para un usuario usando el modelo entrenado"""
    
    if model_knn is None or item_cols is None or user_cols is None:
        raise ValueError("Modelo no cargado correctamente")
    
    if user_id not in user_cols:
        raise ValueError(f"Usuario {user_id} no encontrado en los datos de entrenamiento")
    
    # Cargar la matriz usuario-item completa
    if R_df is None:
        # Recrear la matriz desde los datos
        df = pd.read_csv('data/ratings.csv', 
                       names=['user_id', 'item_id', 'rating', 'timestamp'],
                       sep='\t')
        
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        active_users = user_counts[user_counts >= 5].index
        popular_items = item_counts[item_counts >= 5].index
        df_filtered = df[df['user_id'].isin(active_users) & df['item_id'].isin(popular_items)]
        
        R_df_temp = df_filtered.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    else:
        R_df_temp = R_df
    
    # Obtener índice del usuario en la matriz
    user_idx = user_cols.index(user_id)
    user_vector = R_df_temp.iloc[user_idx:user_idx+1].values
    
    # Encontrar usuarios similares
    distances, indices = model_knn.kneighbors(user_vector, n_neighbors=min(6, len(user_cols)))
    
    # Items que el usuario ya ha calificado
    user_items = set(R_df_temp.columns[R_df_temp.iloc[user_idx] > 0])
    
    # Recopilar recomendaciones de usuarios similares
    recommendations = {}
    
    for neighbor_idx in indices[0][1:]:  # Excluir el propio usuario
        if neighbor_idx < len(user_cols):
            neighbor_user_id = user_cols[neighbor_idx]
            neighbor_row = R_df_temp.loc[neighbor_user_id]
            
            # Items que el vecino ha calificado bien (rating >= 4) y el usuario no ha visto
            for item_id, rating in neighbor_row.items():
                if rating >= 4 and item_id not in user_items:
                    if item_id not in recommendations:
                        recommendations[item_id] = []
                    recommendations[item_id].append(rating)
    
    # Calcular score promedio para cada item recomendado
    item_scores = {}
    for item_id, ratings in recommendations.items():
        item_scores[item_id] = np.mean(ratings)
    
    # Ordenar por score y retornar top N
    sorted_recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Formatear respuesta
    formatted_recommendations = []
    for item_id, score in sorted_recommendations[:n_recommendations]:
        formatted_recommendations.append({
            "item_id": int(item_id),
            "predicted_score": float(score),
            "confidence": "high" if score >= 4.5 else "medium" if score >= 4.0 else "low"
        })
    
    return formatted_recommendations

# --- Endpoints de la API ---

@app.get("/", response_model=HealthResponse)
def health_check():
    """
    Endpoint de verificación del estado de la API
    """
    return HealthResponse(
        status="OK" if model_knn is not None else "ERROR",
        model_loaded=model_knn is not None,
        items_count=len(item_cols) if item_cols else 0,
        users_count=len(user_cols) if user_cols else 0
    )

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend_endpoint(user_id: int, n_recommendations: int = 5):
    """
    Endpoint para obtener recomendaciones personalizadas para un usuario.
    
    - **user_id**: ID del usuario para el cual generar recomendaciones
    - **n_recommendations**: Número de recomendaciones a devolver (default: 5)
    
    Ejemplo de uso: GET /recommend/1?n_recommendations=3
    """
    try:
        recommendations = get_recommendations_for_user(user_id, n_recommendations)
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            message=f"Generadas {len(recommendations)} recomendaciones para usuario {user_id}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

@app.get("/users")
def get_available_users():
    """
    Obtener lista de usuarios disponibles para testing
    """
    if user_cols:
        # Devolver una muestra de usuarios para testing
        sample_users = user_cols[:20] if len(user_cols) > 20 else user_cols
        return {
            "total_users": len(user_cols),
            "sample_users": sample_users,
            "message": f"Total de {len(user_cols)} usuarios disponibles"
        }
    else:
        raise HTTPException(status_code=500, detail="No hay usuarios cargados")

@app.get("/items")
def get_available_items():
    """
    Obtener lista de items disponibles
    """
    if item_cols:
        # Devolver una muestra de items para testing
        sample_items = item_cols[:20] if len(item_cols) > 20 else item_cols
        return {
            "total_items": len(item_cols),
            "sample_items": sample_items,
            "message": f"Total de {len(item_cols)} items disponibles"
        }
    else:
        raise HTTPException(status_code=500, detail="No hay items cargados")

# Información adicional para desarrollo
@app.get("/info")
def api_info():
    """
    Información general sobre la API y el modelo
    """
    return {
        "api_name": "Sistema de Recomendación de Películas",
        "version": "1.0.0",
        "model": "k-Nearest Neighbors (User-Based Collaborative Filtering)",
        "algorithm": "Cosine Similarity",
        "endpoints": {
            "/": "Estado de la API",
            "/recommend/{user_id}": "Obtener recomendaciones",
            "/users": "Lista de usuarios disponibles",
            "/items": "Lista de items disponibles",
            "/info": "Información de la API"
        },
        "model_status": "loaded" if model_knn else "not_loaded"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando API de Recomendaciones...")
    print("📖 Documentación disponible en: http://localhost:8000/docs")
    print("🔍 Estado de la API en: http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)