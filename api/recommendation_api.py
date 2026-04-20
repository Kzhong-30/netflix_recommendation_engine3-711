from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import json
import os
from datetime import datetime
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
from models.ultimate_hybrid_recommender import UltimateHybridRecommender

# Pydantic models for API requests/responses
class UserProfile(BaseModel):
    """User profile for getting recommendations"""
    user_id: int = Field(..., description="Unique user identifier", example=123)
    liked_movies: List[str] = Field(default=[], description="List of movies the user liked", 
                                   example=["Superman", "Mission: Impossible - The Final Reckoning"])
    preferred_genres: List[str] = Field(default=[], description="User's preferred genres", 
                                       example=["Action", "Adventure"])

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: int
    strategy: str
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model_status: str
    uptime_seconds: float

class FeedbackRequest(BaseModel):
    """User feedback on recommendations"""
    user_id: int
    movie_title: str
    action: str = Field(..., description="Action taken", example="clicked|watched|rated")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Rating 1-5 if applicable")

# Global variables
recommender_system = None
start_time = time.time()
request_count = 0
api_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0.0,
    "strategies_used": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    global recommender_system
    print("🚀 Starting Netflix-Style Recommendation API...")
    print("📚 Loading and training ML models...")
    
    try:
        recommender_system = UltimateHybridRecommender()
        success = recommender_system.train()
        
        if success:
            print("✅ ML models loaded successfully!")
            print("🎉 Recommendation API ready to serve!")
        else:
            print("❌ Failed to load ML models")
            raise Exception("Model loading failed")
            
    except Exception as e:
        print(f"💥 Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    print("⏹️  Shutting down Recommendation API...")

# Create FastAPI app
app = FastAPI(
    title="Netflix-Style Movie Recommendation API",
    description="""
    🎬 Production-ready movie recommendation system built with collaborative filtering, 
    content-based filtering, and popularity-based algorithms.
    
    **Features:**
    - Multiple ML algorithms (collaborative filtering, content-based, popularity)
    - Intelligent routing based on user profile
    - Real-time recommendations
    - Performance monitoring
    - User feedback collection
    
    Built by an ML engineer following Netflix's architecture.
    """,
    version="1.0.0",
    contact={
        "name": "ML Engineering Team",
        "email": "ml-team@yourcompany.com"
    },
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Track API usage metrics"""
    global api_metrics, request_count
    
    start_time = time.time()
    request_count += 1
    api_metrics["total_requests"] += 1
    
    try:
        response = await call_next(request)
        
        # Track successful requests
        if response.status_code < 400:
            api_metrics["successful_requests"] += 1
        else:
            api_metrics["failed_requests"] += 1
            
        # Update average response time
        response_time = time.time() - start_time
        current_avg = api_metrics["avg_response_time"]
        total_requests = api_metrics["total_requests"]
        api_metrics["avg_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
        
        return response
        
    except Exception as e:
        api_metrics["failed_requests"] += 1
        raise

def get_recommender():
    """Dependency to get the recommender system"""
    if recommender_system is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    return recommender_system

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎬 Netflix-Style Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "ready"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    global start_time
    
    uptime = time.time() - start_time
    model_status = "ready" if recommender_system and recommender_system.is_trained else "not_ready"
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_status=model_status,
        uptime_seconds=uptime
    )

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    user_profile: UserProfile,
    background_tasks: BackgroundTasks,
    num_recommendations: int = Query(default=10, ge=1, le=50, description="Number of recommendations to return"),
    recommender: UltimateHybridRecommender = Depends(get_recommender)
):
    """
    Get personalized movie recommendations for a user
    
    **Algorithm Selection:**
    - New users (no viewing history): Popularity-based + Genre preferences
    - Users with some data: Hybrid approach (Collaborative + Content-based)
    - Power users (lots of data): Collaborative filtering primary
    
    **Returns:**
    - Personalized movie recommendations
    - Strategy used (for debugging/optimization)
    - Metadata about the recommendation process
    """
    
    try:
        # Convert to dict for recommender system
        profile_dict = {
            "user_id": user_profile.user_id,
            "liked_movies": user_profile.liked_movies,
            "preferred_genres": user_profile.preferred_genres
        }
        
        # Get recommendations from ML system
        start_time = time.time()
        recommendations = recommender.get_smart_recommendations(profile_dict)
        inference_time = time.time() - start_time
        
        # Track strategy usage
        strategy = recommendations.get('strategy', 'unknown')
        if strategy in api_metrics["strategies_used"]:
            api_metrics["strategies_used"][strategy] += 1
        else:
            api_metrics["strategies_used"][strategy] = 1
        
        # Format response
        formatted_recommendations = []
        for section in recommendations.get('sections', []):
            for movie in section.get('movies', [])[:num_recommendations]:
                formatted_recommendations.append({
                    "title": movie.get('title', ''),
                    "vote_average": movie.get('vote_average', 0),
                    "genres": movie.get('genres', []),
                    "overview": movie.get('overview', ''),
                    "recommendation_reason": movie.get('recommendation_reason', section.get('reason', '')),
                    "section": section.get('title', ''),
                    "similarity_score": movie.get('similarity_score'),
                    "predicted_rating": movie.get('predicted_rating'),
                    "collab_score": movie.get('collab_score'),
                    "popularity_score": movie.get('popularity_score')
                })
        
        # Limit to requested number
        formatted_recommendations = formatted_recommendations[:num_recommendations]
        
        # Log recommendation event (in production, send to analytics)
        background_tasks.add_task(
            log_recommendation_event,
            user_profile.user_id,
            strategy,
            len(formatted_recommendations),
            inference_time
        )
        
        return RecommendationResponse(
            user_id=user_profile.user_id,
            strategy=strategy,
            recommendations=formatted_recommendations,
            metadata={
                "total_recommendations": len(formatted_recommendations),
                "inference_time_ms": round(inference_time * 1000, 2),
                "model_version": "1.0.0",
                "algorithms_used": ["collaborative_filtering", "content_based", "popularity_based"]
            },
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"❌ Recommendation error for user {user_profile.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")

@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit user feedback on recommendations
    
    **Use cases:**
    - User clicked on a recommendation (implicit feedback)
    - User watched/rated a recommended movie (explicit feedback)
    - User dismissed a recommendation (negative feedback)
    
    This data is used to improve the recommendation algorithms.
    """
    
    try:
        # Log feedback event (in production, store in database)
        background_tasks.add_task(
            log_feedback_event,
            feedback.user_id,
            feedback.movie_title,
            feedback.action,
            feedback.rating
        )
        
        return {
            "message": "Feedback received successfully",
            "user_id": feedback.user_id,
            "movie": feedback.movie_title,
            "action": feedback.action
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")

@app.get("/metrics")
async def get_api_metrics():
    """
    Get API performance metrics
    
    **For monitoring and optimization:**
    - Request counts and success rates
    - Average response times
    - Strategy usage distribution
    - System uptime
    """
    
    global api_metrics, start_time
    
    return {
        "api_metrics": api_metrics,
        "uptime_seconds": time.time() - start_time,
        "total_requests": request_count,
        "system_status": "healthy" if recommender_system else "unhealthy"
    }



@app.get("/movies")
async def list_movies(
    limit: int = Query(default=20, ge=1, le=100),
    genre: Optional[str] = None,
    recommender: UltimateHybridRecommender = Depends(get_recommender)
):
    """
    List available movies in the system
    
    **Useful for:**
    - Frontend autocomplete
    - User profile building
    - Content discovery
    """
    
    try:
        movies_df = recommender.content_recommender.movies_df
        
        if genre:
            # Filter by genre
            filtered_movies = []
            for _, movie in movies_df.iterrows():
                movie_genres = movie['genres']
                if isinstance(movie_genres, str):
                    movie_genres = eval(movie_genres)
                
                if genre in movie_genres:
                    filtered_movies.append(movie)
            
            movies_df = pd.DataFrame(filtered_movies)
        
        # Sort by popularity and limit
        movies_df = movies_df.nlargest(limit, 'vote_average')
        
        movies_list = []
        for _, movie in movies_df.iterrows():
            movies_list.append({
                "title": movie['title'],
                "vote_average": movie['vote_average'],
                "genres": movie['genres'],
                "release_date": movie['release_date'],
                "overview": movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else movie['overview']
            })
        
        return {
            "movies": movies_list,
            "total_count": len(movies_list),
            "filter": {"genre": genre} if genre else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve movies: {str(e)}")

# Background task functions
async def log_recommendation_event(user_id: int, strategy: str, num_recs: int, inference_time: float):
    """Log recommendation events for analytics"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "recommendation",
        "user_id": user_id,
        "strategy": strategy,
        "num_recommendations": num_recs,
        "inference_time_ms": round(inference_time * 1000, 2)
    }
    
    # In production, send to your analytics system (e.g., ElasticSearch, DataDog)
    print(f"📊 Recommendation Event: {json.dumps(event)}")

async def log_feedback_event(user_id: int, movie_title: str, action: str, rating: Optional[float]):
    """Log user feedback for model improvement"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "feedback",
        "user_id": user_id,
        "movie_title": movie_title,
        "action": action,
        "rating": rating
    }
    
    # In production, store in your feedback database for model retraining
    print(f"👍 Feedback Event: {json.dumps(event)}")

# Run the API
if __name__ == "__main__":
    print("🚀 Starting Netflix-Style Recommendation API...")
    uvicorn.run(
        "recommendation_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )