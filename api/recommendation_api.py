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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import cold_start_handler, ColdStartRecommender, ContentRecommender

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

class ColdStartRequest(BaseModel):
    """Cold start recommendation request"""
    user_id: int = Field(..., description="Unique user identifier", example=456)
    preferred_genres: List[str] = Field(..., description="User's preferred genres from registration", 
                                       example=["Action", "Sci-Fi", "Drama"])
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    lambda_param: float = Field(default=0.6, ge=0.0, le=1.0, 
                                description="MMR diversity parameter (0=diversity only, 1=relevance only)")

class ColdStartResponse(BaseModel):
    """Cold start recommendation response"""
    user_id: int
    strategy: str
    interaction_count: int
    is_new_user: bool
    preferred_genres: List[str]
    recommendations: List[Dict[str, Any]]
    diversity_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    accuracy_constraint_met: bool
    timestamp: datetime

class UserInteractionRequest(BaseModel):
    """User interaction tracking request"""
    user_id: int = Field(..., description="User identifier")
    interaction_type: str = Field(default="view", description="Type of interaction (view, click, rate)")

class UserStatusResponse(BaseModel):
    """User cold start status response"""
    user_id: int
    interaction_count: int
    is_in_cold_start: bool
    interactions_to_exit: int

# Global variables
recommender_system = None
content_recommender = None
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
    global recommender_system, content_recommender
    print("🚀 Starting Netflix-Style Recommendation API...")
    print("📚 Loading and training ML models...")
    
    try:
        content_recommender = ContentRecommender()
        recommender_system = ColdStartRecommender()
        
        print("✅ ML models loaded successfully!")
        print("🎉 Cold Start Recommendation API ready to serve!")
            
    except Exception as e:
        print(f"💥 Startup error: {e}")
        raise
    
    yield
    
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
    if content_recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    return content_recommender

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
    num_recommendations: int = Query(default=10, ge=1, le=50, description="Number of recommendations to return")
):
    """
    Get personalized movie recommendations for a user using cold start system
    
    **Algorithm Selection:**
    - New users (no viewing history): Genre-based + MMR diversity injection
    - Uses TF-IDF content features + Maximal Marginal Relevance algorithm
    
    **Returns:**
    - Personalized movie recommendations
    - Strategy used (for debugging/optimization)
    - Metadata about the recommendation process
    """
    
    try:
        # Use cold start recommendation system
        start_time = time.time()
        cold_start_result = cold_start_handler.get_cold_start_recommendations(
            user_id=user_profile.user_id,
            preferred_genres=user_profile.preferred_genres,
            num_recommendations=num_recommendations
        )
        inference_time = time.time() - start_time
        
        strategy = cold_start_result['strategy']
        if strategy in api_metrics["strategies_used"]:
            api_metrics["strategies_used"][strategy] += 1
        else:
            api_metrics["strategies_used"][strategy] = 1
        
        formatted_recommendations = []
        for movie in cold_start_result['recommendations'][:num_recommendations]:
            formatted_recommendations.append({
                "title": movie.get('title', ''),
                "vote_average": movie.get('vote_average', 0),
                "genres": movie.get('genres', []),
                "overview": movie.get('overview', ''),
                "recommendation_reason": movie.get('recommendation_reason', ''),
                "section": "Cold Start Recommendations",
                "similarity_score": movie.get('match_score'),
                "predicted_rating": movie.get('vote_average'),
                "popularity_score": movie.get('popularity')
            })
        
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
                "algorithms_used": ["content_based_tfidf", "mmr_diversity", "popularity_based"],
                "diversity_metrics": cold_start_result['diversity_metrics'],
                "accuracy_estimate": cold_start_result['performance_metrics']['accuracy_estimate_percent']
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
    recommender: ContentRecommender = Depends(get_recommender)
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

@app.post("/api/recommend/cold-start", response_model=ColdStartResponse, tags=["Cold Start"])
async def cold_start_recommendation(request: ColdStartRequest):
    """
    🆕 **Cold Start Recommendation Endpoint**
    
    Get initial movie recommendations for NEW USERS based on their genre preferences
    selected during registration. Uses MMR (Maximal Marginal Relevance) algorithm
    to ensure diversity while maintaining recommendation accuracy.
    
    **Features:**
    - TF-IDF based content feature extraction
    - MMR diversity injection (configurable lambda parameter)
    - Genre matching with quality scoring
    - Guaranteed < 500ms response time
    
    **Constraints Met:**
    - Accuracy > 40% for first 10 recommendations
    - Genre repetition < 30% in recommendation list
    - Response time < 500ms
    
    **Parameters:**
    - `user_id`: Unique user identifier
    - `preferred_genres`: List of movie genres the user likes (from registration)
    - `num_recommendations`: Number of movies to recommend (1-50)
    - `lambda_param`: MMR diversity trade-off (0 = max diversity, 1 = max relevance)
    
    **Recommended Lambda Values:**
    - 0.6: Default, balanced relevance and diversity ✓
    - 0.3: High diversity, good for exploration
    - 0.9: High relevance, good for known preferences
    """
    try:
        result = cold_start_handler.get_cold_start_recommendations(
            user_id=request.user_id,
            preferred_genres=request.preferred_genres,
            num_recommendations=request.num_recommendations,
            lambda_param=request.lambda_param
        )
        
        result['timestamp'] = datetime.now()
        
        return result
        
    except Exception as e:
        print(f"❌ Cold start recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate cold start recommendations: {str(e)}")

@app.post("/api/recommend/cold-start/interaction", tags=["Cold Start"])
async def record_interaction(request: UserInteractionRequest):
    """
    Record user interaction to track transition out of cold start phase.
    
    After 10 interactions, user exits cold start phase and transitions
    to hybrid recommendation strategy.
    """
    try:
        cold_start_handler.record_user_interaction(request.user_id)
        status = cold_start_handler.get_user_status(request.user_id)
        
        return {
            "message": "Interaction recorded",
            "user_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")

@app.get("/api/recommend/cold-start/status/{user_id}", response_model=UserStatusResponse, tags=["Cold Start"])
async def get_user_cold_start_status(user_id: int):
    """Check if a user is still in cold start phase"""
    try:
        return cold_start_handler.get_user_status(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user status: {str(e)}")

@app.get("/api/recommend/cold-start/info", tags=["Cold Start"])
async def get_cold_start_info():
    """Get information about the cold start recommendation system"""
    return {
        "algorithm": "MMR (Maximal Marginal Relevance) + TF-IDF Content Features",
        "feature_extraction": "scikit-learn TfidfVectorizer",
        "constraints": {
            "target_accuracy": "> 40%",
            "max_genre_repetition": "< 30%",
            "max_response_time": "< 500ms"
        },
        "cold_start_threshold": "10 interactions",
        "lambda_parameter_range": "0.0 (diversity) to 1.0 (relevance)",
        "default_lambda": 0.6
    }

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