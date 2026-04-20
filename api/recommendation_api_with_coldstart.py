from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import json
import os
import sys
from datetime import datetime
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cold_start import (
    ColdStartRecommender, 
    initialize_cold_start_recommender, 
    get_cold_start_recommender,
    inject_diversity
)

# Pydantic models for API requests/responses
class UserProfile(BaseModel):
    """User profile for getting recommendations"""
    user_id: int = Field(..., description="Unique user identifier", example=123)
    liked_movies: List[str] = Field(default=[], description="List of movies the user liked", 
                                   example=["Superman", "Mission: Impossible - The Final Reckoning"])
    preferred_genres: List[str] = Field(default=[], description="User's preferred genres", 
                                       example=["Action", "Adventure"])

class ColdStartRequest(BaseModel):
    """冷启动推荐请求模型"""
    user_id: int = Field(..., description="新用户唯一标识", example=10001)
    genres: List[str] = Field(
        ..., 
        description="用户注册时选择的类型偏好", 
        example=["Action", "Sci-Fi", "Thriller"],
        min_items=1,
        max_items=5
    )
    num_recommendations: int = Field(
        default=10,
        ge=1,
        le=20,
        description="需要获取的推荐数量"
    )

class ColdStartMovieResponse(BaseModel):
    """冷启动推荐电影响应模型"""
    id: int
    title: str
    overview: str
    genres: List[str]
    vote_average: float
    popularity: float
    release_date: str

class ColdStartRecommendationItem(BaseModel):
    """单个冷启动推荐项"""
    movie: ColdStartMovieResponse
    relevance_score: float = Field(..., description="相关性分数 (0-1)")
    diversity_score: float = Field(..., description="多样性分数")
    final_score: float = Field(..., description="最终综合分数")
    reason: str = Field(..., description="推荐理由")

class ColdStartResponse(BaseModel):
    """冷启动推荐响应模型"""
    user_id: int
    recommendations: List[ColdStartRecommendationItem]
    metadata: Dict[str, Any]
    timestamp: datetime

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
    cold_start_ready: bool
    uptime_seconds: float

class FeedbackRequest(BaseModel):
    """User feedback on recommendations"""
    user_id: int
    movie_title: str
    action: str = Field(..., description="Action taken", example="clicked|watched|rated")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Rating 1-5 if applicable")

# Global variables
cold_start_recommender = None
start_time = time.time()
request_count = 0
api_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_response_time": 0.0,
    "cold_start_requests": 0,
    "strategies_used": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global cold_start_recommender
    
    # Startup
    print("🚀 Starting Netflix-Style Recommendation API with Cold Start Support...")
    print("📚 Initializing cold start recommender...")
    
    try:
        # 加载电影数据
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        movies_file = os.path.join(data_path, 'tmdb_5000_movies.csv')
        
        if os.path.exists(movies_file):
            movies_df = pd.read_csv(movies_file)
            cold_start_recommender = initialize_cold_start_recommender(movies_df)
            print(f"✅ Cold start recommender initialized with {len(movies_df)} movies!")
        else:
            print(f"⚠️ Movies data file not found at {movies_file}")
            print("⚠️ Cold start recommender will use mock data")
            # 创建模拟数据用于测试
            mock_data = {
                'title': ['The Dark Knight', 'Inception', 'Avatar', 'The Avengers', 'Titanic',
                         'The Matrix', 'Interstellar', 'Pulp Fiction', 'The Godfather', 'Forrest Gump',
                         'Fight Club', 'The Lord of the Rings', 'Star Wars', 'Gladiator', 'The Shawshank Redemption'],
                'overview': [
                    'Batman faces the Joker in Gotham City',
                    'A thief who steals corporate secrets through dream-sharing technology',
                    'A paraplegic marine dispatched to the moon Pandora',
                    'Earth mightiest heroes must come together to save the world',
                    'A seventeen-year-old aristocrat falls in love with a kind but poor artist',
                    'A computer hacker learns about the true nature of reality',
                    'A team of explorers travel through a wormhole in space',
                    'The lives of two mob hitmen, a boxer, and others intertwine',
                    'The aging patriarch of an organized crime dynasty transfers control',
                    'The presidencies of Kennedy and Johnson, Vietnam, Watergate',
                    'An insomniac office worker and a devil-may-care soapmaker',
                    'A meek Hobbit from the Shire and eight companions set out on a journey',
                    'Luke Skywalker joins forces with a Jedi Knight',
                    'A former Roman General sets out to exact vengeance',
                    'Two imprisoned men bond over a number of years'
                ],
                'genres': [
                    ['Action', 'Crime', 'Drama'], ['Action', 'Sci-Fi', 'Thriller'], 
                    ['Action', 'Adventure', 'Sci-Fi'], ['Action', 'Adventure', 'Sci-Fi'],
                    ['Drama', 'Romance'], ['Action', 'Sci-Fi'],
                    ['Adventure', 'Drama', 'Sci-Fi'], ['Crime', 'Drama'],
                    ['Crime', 'Drama'], ['Drama', 'Romance'],
                    ['Drama'], ['Adventure', 'Fantasy'],
                    ['Action', 'Adventure', 'Fantasy'], ['Action', 'Adventure', 'Drama'],
                    ['Drama']
                ],
                'vote_average': [9.0, 8.8, 7.8, 8.0, 7.9, 8.7, 8.6, 8.9, 9.2, 8.8, 8.8, 8.9, 8.6, 8.5, 9.3],
                'popularity': [100.0, 95.0, 150.0, 120.0, 80.0, 90.0, 85.0, 88.0, 92.0, 87.0, 86.0, 110.0, 130.0, 89.0, 91.0],
                'release_date': ['2008-07-18', '2010-07-16', '2009-12-18', '2012-05-04', '1997-12-19',
                               '1999-03-31', '2014-11-07', '1994-10-14', '1972-03-24', '1994-07-06',
                               '1999-10-15', '2001-12-19', '1977-05-25', '2000-05-05', '1994-10-14']
            }
            movies_df = pd.DataFrame(mock_data)
            cold_start_recommender = initialize_cold_start_recommender(movies_df)
            print(f"✅ Cold start recommender initialized with mock data ({len(movies_df)} movies)!")
        
        print("🎉 Recommendation API ready to serve!")
            
    except Exception as e:
        print(f"💥 Startup error: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Shutdown
    print("⏹️  Shutting down Recommendation API...")

# Create FastAPI app
app = FastAPI(
    title="Netflix-Style Movie Recommendation API",
    description="""
    🎬 Production-ready movie recommendation system with Cold Start support.
    
    **Features:**
    - Cold Start recommendations for new users
    - TF-IDF content feature extraction
    - MMR (Maximal Marginal Relevance) diversity injection
    - Real-time recommendations
    - Performance monitoring
    
    **Cold Start API:**
    - `POST /api/recommend/cold-start` - Get initial recommendations for new users
    """,
    version="1.1.0",
    contact={
        "name": "ML Engineering Team",
        "email": "ml-team@yourcompany.com"
    },
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Track API usage metrics"""
    global api_metrics, request_count
    
    start_time_req = time.time()
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
        response_time = time.time() - start_time_req
        current_avg = api_metrics["avg_response_time"]
        total_requests = api_metrics["total_requests"]
        api_metrics["avg_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
        
        return response
        
    except Exception as e:
        api_metrics["failed_requests"] += 1
        raise

def get_recommender():
    """Dependency to get the cold start recommender system"""
    if cold_start_recommender is None:
        raise HTTPException(status_code=503, detail="Cold start recommendation system not available")
    return cold_start_recommender

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🎬 Netflix-Style Movie Recommendation API with Cold Start Support",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "cold_start": "/api/recommend/cold-start",
        "status": "ready"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    global start_time
    
    uptime = time.time() - start_time
    is_ready = cold_start_recommender is not None and cold_start_recommender.is_trained
    
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        version="1.1.0",
        cold_start_ready=is_ready,
        uptime_seconds=uptime
    )

@app.post("/api/recommend/cold-start", response_model=ColdStartResponse)
async def get_cold_start_recommendations(
    request: ColdStartRequest,
    background_tasks: BackgroundTasks,
    recommender: ColdStartRecommender = Depends(get_recommender)
):
    """
    冷启动推荐端点 - 为新用户提供初始推荐
    
    **功能特点：**
    - 基于用户类型偏好的内容匹配
    - TF-IDF内容特征提取
    - MMR多样性注入算法
    - 响应时间 < 500ms
    
    **请求示例：**
    ```json
    {
        "user_id": 10001,
        "genres": ["Action", "Sci-Fi", "Thriller"],
        "num_recommendations": 10
    }
    ```
    
    **返回说明：**
    - `relevance_score`: 相关性分数 (0-1)，越高表示越符合用户偏好
    - `diversity_score`: 多样性分数
    - `final_score`: 最终综合分数
    - `reason`: 推荐理由说明
    
    **验收指标：**
    - 新用户前10次推荐准确率 > 40%
    - 推荐列表中重复类型不超过 30%
    - 响应时间 < 500ms
    """
    global api_metrics
    
    start_time_req = time.time()
    
    try:
        # 调用冷启动推荐
        result = recommender.get_cold_start_recommendations(
            preferred_genres=request.genres,
            num_recommendations=request.num_recommendations,
            diversity_lambda=0.5
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Recommendation failed"))
        
        # 计算响应时间
        response_time = time.time() - start_time_req
        
        # 更新指标
        api_metrics["cold_start_requests"] += 1
        
        # 评估推荐质量（用于监控）
        quality_metrics = recommender.evaluate_recommendation_quality(
            result["recommendations"],
            request.genres
        )
        
        # 后台记录
        background_tasks.add_task(
            log_cold_start_event,
            request.user_id,
            len(result["recommendations"]),
            response_time,
            quality_metrics
        )
        
        # 检查性能要求
        if response_time > 0.5:  # 500ms
            print(f"⚠️ Cold start response time warning: {response_time*1000:.2f}ms")
        
        return ColdStartResponse(
            user_id=request.user_id,
            recommendations=result["recommendations"],
            metadata={
                **result["metadata"],
                "quality_metrics": quality_metrics,
                "response_time_ms": round(response_time * 1000, 2)
            },
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Cold start recommendation error for user {request.user_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate cold start recommendations: {str(e)}")

@app.get("/metrics")
async def get_api_metrics():
    """
    Get API performance metrics
    
    **For monitoring and optimization:**
    - Request counts and success rates
    - Average response times
    - Cold start usage statistics
    - System uptime
    """
    global api_metrics, start_time
    
    return {
        "api_metrics": api_metrics,
        "uptime_seconds": time.time() - start_time,
        "total_requests": request_count,
        "cold_start_status": "ready" if cold_start_recommender else "unavailable"
    }

@app.get("/movies")
async def list_movies(
    limit: int = Query(default=20, ge=1, le=100),
    genre: Optional[str] = None
):
    """
    List available movies in the system
    
    **Useful for:**
    - Frontend autocomplete
    - User profile building
    - Content discovery
    """
    
    try:
        if cold_start_recommender is None or not cold_start_recommender.movies:
            return {"movies": [], "total_count": 0}
        
        movies = cold_start_recommender.movies
        
        if genre:
            # Filter by genre
            movies = [m for m in movies if genre in m.genres]
        
        # Sort by vote_average and limit
        movies = sorted(movies, key=lambda m: m.vote_average, reverse=True)[:limit]
        
        movies_list = []
        for movie in movies:
            movies_list.append({
                "id": movie.id,
                "title": movie.title,
                "vote_average": movie.vote_average,
                "genres": movie.genres,
                "release_date": movie.release_date,
                "overview": movie.overview[:200] + "..." if len(movie.overview) > 200 else movie.overview
            })
        
        return {
            "movies": movies_list,
            "total_count": len(movies_list),
            "filter": {"genre": genre} if genre else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve movies: {str(e)}")

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

# Background task functions
async def log_cold_start_event(
    user_id: int, 
    num_recs: int, 
    inference_time: float,
    quality_metrics: Dict[str, float]
):
    """Log cold start recommendation events for analytics"""
    event = {
        "timestamp": datetime.now().isoformat(),
        "event_type": "cold_start_recommendation",
        "user_id": user_id,
        "num_recommendations": num_recs,
        "inference_time_ms": round(inference_time * 1000, 2),
        "accuracy": quality_metrics.get("accuracy"),
        "genre_diversity": quality_metrics.get("genre_diversity"),
        "avg_rating": quality_metrics.get("avg_rating")
    }
    
    print(f"📊 Cold Start Event: {json.dumps(event)}")

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
    print("🚀 Starting Netflix-Style Recommendation API with Cold Start Support...")
    uvicorn.run(
        "api.recommendation_api_with_coldstart:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
