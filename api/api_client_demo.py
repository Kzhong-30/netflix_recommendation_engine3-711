import requests
import json
from typing import List, Optional, Dict

class RecommendationAPIClient:
    """
    Python client for the Netflix-Style Recommendation API
    
    Makes it easy to integrate recommendations into any application
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def get_recommendations(self, user_id: int, liked_movies: List[str] = None, 
                          preferred_genres: List[str] = None, 
                          num_recommendations: int = 10) -> Dict:
        """Get personalized movie recommendations"""
        
        payload = {
            "user_id": user_id,
            "liked_movies": liked_movies or [],
            "preferred_genres": preferred_genres or []
        }
        
        params = {"num_recommendations": num_recommendations}
        
        try:
            response = self.session.post(
                f"{self.base_url}/recommendations",
                json=payload,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return {"error": str(e)}
    
    def submit_feedback(self, user_id: int, movie_title: str, 
                       action: str, rating: Optional[float] = None) -> Dict:
        """Submit user feedback on recommendations"""
        
        payload = {
            "user_id": user_id,
            "movie_title": movie_title,
            "action": action
        }
        
        if rating:
            payload["rating"] = rating
            
        try:
            response = self.session.post(
                f"{self.base_url}/feedback",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Feedback submission failed: {e}")
            return {"error": str(e)}
    
    def get_movies(self, limit: int = 20, genre: Optional[str] = None) -> Dict:
        """Get list of available movies"""
        
        params = {"limit": limit}
        if genre:
            params["genre"] = genre
            
        try:
            response = self.session.get(
                f"{self.base_url}/movies",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Movies request failed: {e}")
            return {"error": str(e)}
    
    def get_health(self) -> Dict:
        """Check API health status"""
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return {"error": str(e)}
    
    def get_metrics(self) -> Dict:
        """Get API performance metrics"""
        
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=5)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Metrics request failed: {e}")
            return {"error": str(e)}

def demo_api_client():
    """Demo the API client with real requests"""
    print("🎬 TESTING NETFLIX-STYLE RECOMMENDATION API")
    print("="*60)
    
    # Initialize client
    client = RecommendationAPIClient()
    
    # 1. Health check
    print("🏥 Checking API health...")
    health = client.get_health()
    if "error" not in health:
        print(f"✅ API Status: {health.get('status')}")
        print(f"   Model Status: {health.get('model_status')}")
        print(f"   Uptime: {health.get('uptime_seconds', 0):.1f} seconds")
    else:
        print("❌ API is not running! Start it first with: python api/recommendation_api.py")
        return
    
    # 2. Get recommendations for different user types
    test_users = [
        {
            "user_id": 1,
            "liked_movies": ["Superman", "Mission: Impossible - The Final Reckoning"],
            "preferred_genres": ["Action"],
            "description": "Action movie fan"
        },
        {
            "user_id": 999,  # New user
            "liked_movies": [],
            "preferred_genres": ["Comedy", "Drama"],
            "description": "New user who likes Comedy & Drama"
        }
    ]
    
    for user in test_users:
        print(f"\n🎯 Getting recommendations for: {user['description']}")
        print(f"   User ID: {user['user_id']}")
        print(f"   Liked movies: {user['liked_movies']}")
        print(f"   Preferred genres: {user['preferred_genres']}")
        
        recommendations = client.get_recommendations(
            user_id=user["user_id"],
            liked_movies=user["liked_movies"],
            preferred_genres=user["preferred_genres"],
            num_recommendations=5
        )
        
        if "error" not in recommendations:
            print(f"   Strategy used: {recommendations.get('strategy')}")
            print(f"   Found {len(recommendations.get('recommendations', []))} movies:")
            
            for i, rec in enumerate(recommendations.get('recommendations', [])[:3], 1):
                print(f"     {i}. {rec.get('title')} (Rating: {rec.get('vote_average', 0):.1f})")
                print(f"        Reason: {rec.get('recommendation_reason', 'N/A')}")
        else:
            print(f"   ❌ Error: {recommendations['error']}")
    
    # 3. Submit feedback
    print(f"\n👍 Submitting feedback...")
    feedback_result = client.submit_feedback(
        user_id=1,
        movie_title="Superman",
        action="watched",
        rating=4.5
    )
    
    if "error" not in feedback_result:
        print(f"✅ Feedback submitted: {feedback_result.get('message')}")
    else:
        print(f"❌ Feedback error: {feedback_result['error']}")
    
    # 4. Get API metrics
    print(f"\n📊 API Performance Metrics:")
    metrics = client.get_metrics()
    
    if "error" not in metrics:
        api_metrics = metrics.get('api_metrics', {})
        print(f"   Total requests: {api_metrics.get('total_requests', 0)}")
        print(f"   Success rate: {api_metrics.get('successful_requests', 0)}/{api_metrics.get('total_requests', 0)}")
        print(f"   Avg response time: {api_metrics.get('avg_response_time', 0)*1000:.1f}ms")
        print(f"   Strategies used: {api_metrics.get('strategies_used', {})}")
    
    # 5. Browse movies
    print(f"\n🎬 Available Action Movies:")
    movies = client.get_movies(limit=5, genre="Action")
    
    if "error" not in movies:
        for movie in movies.get('movies', []):
            print(f"   • {movie.get('title')} (Rating: {movie.get('vote_average', 0):.1f})")

if __name__ == "__main__":
    demo_api_client()