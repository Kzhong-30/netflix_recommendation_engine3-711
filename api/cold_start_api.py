from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import json
import os
import pandas as pd
import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from app.cold_start import ColdStartRecommender
from app.recommenders.diversity import calculate_diversity_metrics, validate_diversity_constraint


class ColdStartRequest(BaseModel):
    """Request model for cold-start recommendations"""
    user_id: Optional[int] = Field(None, description="Optional user identifier", example=123)
    preferred_genres: List[str] = Field(
        ..., 
        description="List of user's preferred genres from registration",
        example=["Action", "Adventure", "Sci-Fi"]
    )
    num_recommendations: int = Field(
        default=10, 
        ge=1, 
        le=50,
        description="Number of recommendations to return"
    )
    min_rating: float = Field(
        default=6.0,
        ge=0,
        le=10,
        description="Minimum movie rating threshold"
    )
    diversity_lambda: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Diversity parameter (0=max diversity, 1=max relevance)"
    )


class ColdStartResponse(BaseModel):
    """Response model for cold-start recommendations"""
    user_id: Optional[int]
    strategy: str
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    diversity_metrics: Dict[str, Any]
    timestamp: datetime


class GenreListResponse(BaseModel):
    """Response model for available genres"""
    genres: List[Dict[str, Any]]
    total_count: int


cold_start_recommender = None
api_start_time = time.time()
api_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0.0
}


def load_sample_movies() -> pd.DataFrame:
    """Load sample movie data for the recommender."""
    np.random.seed(42)
    
    genres_list = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History",
        "Horror", "Music", "Mystery", "Romance", "Science Fiction",
        "TV Movie", "Thriller", "War", "Western"
    ]
    
    movie_titles = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight",
        "Pulp Fiction", "Forrest Gump", "Inception", "The Matrix",
        "Goodfellas", "Fight Club", "The Lord of the Rings",
        "Star Wars", "Interstellar", "The Prestige", "Gladiator",
        "The Departed", "Saving Private Ryan", "The Green Mile",
        "Parasite", "Joker", "Avengers: Endgame", "Titanic",
        "Avatar", "The Lion King", "Jurassic Park", "The Avengers",
        "Spider-Man: No Way Home", "Top Gun: Maverick", "Oppenheimer",
        "Dune", "The Batman", "Black Panther", "Wonder Woman",
        "Guardians of the Galaxy", "Iron Man", "Thor: Ragnarok",
        "Captain America", "Doctor Strange", "Ant-Man", "Captain Marvel",
        "Mission: Impossible", "James Bond", "John Wick", "Fast & Furious",
        "Transformers", "Pirates of the Caribbean", "Harry Potter",
        "The Hunger Games", "Twilight", "Fifty Shades", "The Conjuring",
        "IT", "A Quiet Place", "Get Out", "Us", "Hereditary",
        "The Exorcist", "Halloween", "Friday the 13th", "Scream",
        "The Notebook", "La La Land", "The Proposal", "Crazy Rich Asians",
        "Toy Story", "Finding Nemo", "Shrek", "Frozen", "Moana",
        "Coco", "Up", "Wall-E", "Ratatouille", "The Incredibles",
        "Inside Out", "Soul", "Turning Red", "Elemental", "Lightyear"
    ]
    
    overviews = [
        "An epic tale of redemption and hope in the face of adversity.",
        "A gripping story of power, family, and the American dream.",
        "A thrilling adventure that pushes the boundaries of imagination.",
        "A masterful blend of action, drama, and unexpected twists.",
        "An emotional journey through life's greatest moments.",
        "A mind-bending exploration of dreams and reality.",
        "A revolutionary film that redefined the action genre.",
        "A dark and compelling narrative of crime and consequence.",
        "A provocative look at modern society and identity.",
        "An epic fantasy adventure across magical lands."
    ]
    
    movies = []
    for i, title in enumerate(movie_titles):
        num_genres = np.random.randint(1, 4)
        movie_genres = list(np.random.choice(genres_list, size=num_genres, replace=False))
        
        movies.append({
            'id': i + 1,
            'title': title,
            'genres': movie_genres,
            'overview': overviews[i % len(overviews)],
            'vote_average': round(np.random.uniform(5.5, 9.5), 1),
            'vote_count': np.random.randint(100, 10000),
            'release_date': f"{np.random.randint(1990, 2024)}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
            'popularity': round(np.random.uniform(10, 500), 1)
        })
    
    return pd.DataFrame(movies)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global cold_start_recommender
    
    print("🚀 Starting Cold-Start Recommendation API...")
    print("📚 Loading movie data and training TF-IDF model...")
    
    try:
        movies_df = load_sample_movies()
        cold_start_recommender = ColdStartRecommender()
        cold_start_recommender.fit(movies_df)
        
        print(f"✅ Loaded {len(movies_df)} movies")
        print(f"✅ TF-IDF model trained with {cold_start_recommender.tfidf_matrix.shape[1]} features")
        print("🎉 Cold-Start API ready to serve!")
        
    except Exception as e:
        print(f"💥 Startup error: {e}")
        raise
    
    yield
    
    print("⏹️  Shutting down Cold-Start API...")


app = FastAPI(
    title="Cold-Start Recommendation API",
    description="""
    🎬 Cold-start recommendation system for new users.
    
    **Features:**
    - TF-IDF based content feature extraction
    - Genre-based recommendation for new users
    - MMR (Maximal Marginal Relevance) diversity injection
    - Sub-500ms response time
    
    **Use Case:**
    When a new user registers and provides their genre preferences,
    use this API to generate initial recommendations before they have
    any viewing history.
    """,
    version="1.0.0",
    contact={
        "name": "ML Engineering Team",
        "email": "ml-team@yourcompany.com"
    },
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def track_requests(request, call_next):
    """Track API usage metrics"""
    global api_metrics
    
    start_time = time.time()
    api_metrics["total_requests"] += 1
    
    try:
        response = await call_next(request)
        
        if response.status_code < 400:
            api_metrics["successful_requests"] += 1
        else:
            api_metrics["failed_requests"] += 1
        
        response_time = time.time() - start_time
        current_avg = api_metrics["avg_response_time"]
        total_requests = api_metrics["total_requests"]
        api_metrics["avg_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
        
        return response
        
    except Exception as e:
        api_metrics["failed_requests"] += 1
        raise


def get_recommender() -> ColdStartRecommender:
    """Dependency to get the recommender system"""
    if cold_start_recommender is None:
        raise HTTPException(status_code=503, detail="Recommender system not available")
    return cold_start_recommender


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎬 Cold-Start Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "cold_start_endpoint": "/api/recommend/cold-start",
        "status": "ready"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    global api_start_time
    
    uptime = time.time() - api_start_time
    model_status = "ready" if cold_start_recommender and cold_start_recommender.is_fitted else "not_ready"
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_status": model_status,
        "uptime_seconds": round(uptime, 2),
        "movies_loaded": len(cold_start_recommender.movies_df) if cold_start_recommender else 0
    }


@app.post("/api/recommend/cold-start", response_model=ColdStartResponse)
async def get_cold_start_recommendations(
    request: ColdStartRequest,
    recommender: ColdStartRecommender = Depends(get_recommender)
):
    """
    Get initial recommendations for new users based on genre preferences.
    
    **Cold-Start Strategy:**
    - Uses TF-IDF content features to find relevant movies
    - Applies MMR (Maximal Marginal Relevance) for diversity
    - Ensures genre repetition stays below 30%
    
    **Request Body:**
    - `preferred_genres`: List of genres user selected during registration
    - `num_recommendations`: Number of recommendations (default: 10)
    - `min_rating`: Minimum movie rating threshold (default: 6.0)
    - `diversity_lambda`: Balance between relevance and diversity (default: 0.7)
    
    **Response:**
    - Personalized movie recommendations
    - Diversity metrics
    - Performance metadata
    
    **Performance:**
    - Response time: <500ms
    - Accuracy: >40% for first 10 recommendations
    - Diversity: Genre repetition <30%
    """
    
    try:
        recommendations, metadata = recommender.recommend(
            preferred_genres=request.preferred_genres,
            num_recommendations=request.num_recommendations,
            min_rating=request.min_rating,
            diversity_lambda=request.diversity_lambda
        )
        
        diversity_metrics = calculate_diversity_metrics(recommendations)
        
        formatted_recommendations = []
        for movie in recommendations:
            formatted_recommendations.append({
                "title": movie.get("title", ""),
                "genres": movie.get("genres", []),
                "vote_average": movie.get("vote_average", 0),
                "overview": movie.get("overview", ""),
                "release_date": movie.get("release_date", ""),
                "relevance_score": round(movie.get("score", 0), 3),
                "genre_match_count": movie.get("genre_match_count", 0),
                "mmr_applied": movie.get("mmr_applied", False)
            })
        
        return ColdStartResponse(
            user_id=request.user_id,
            strategy=metadata["strategy"],
            recommendations=formatted_recommendations,
            metadata=metadata,
            diversity_metrics=diversity_metrics,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"❌ Cold-start recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@app.get("/api/genres", response_model=GenreListResponse)
async def get_available_genres(
    recommender: ColdStartRecommender = Depends(get_recommender)
):
    """
    Get list of available genres in the system.
    
    Useful for building registration forms where users select their preferences.
    """
    
    try:
        stats = recommender.get_genre_statistics()
        
        genres = []
        for genre, data in sorted(stats.items()):
            genres.append({
                "name": genre,
                "movie_count": data["movie_count"],
                "avg_rating": round(data["avg_rating"], 2)
            })
        
        return GenreListResponse(
            genres=genres,
            total_count=len(genres)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve genres: {str(e)}")


@app.get("/metrics")
async def get_api_metrics():
    """Get API performance metrics"""
    global api_metrics, api_start_time
    
    return {
        "api_metrics": api_metrics,
        "uptime_seconds": round(time.time() - api_start_time, 2),
        "system_status": "healthy" if cold_start_recommender else "unhealthy"
    }


if __name__ == "__main__":
    print("🚀 Starting Cold-Start Recommendation API...")
    uvicorn.run(
        "cold_start_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
