import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting Netflix-Style Cold Start Recommendation API...")
    uvicorn.run(
        "api.recommendation_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
