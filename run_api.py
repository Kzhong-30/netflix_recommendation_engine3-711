"""
冷启动推荐API启动脚本

用法:
    python run_api.py

API将在 http://localhost:8000 启动
文档地址: http://localhost:8000/docs
"""

import uvicorn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Starting Netflix-Style Recommendation API with Cold Start Support...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("❄️  Cold Start Endpoint: POST http://localhost:8000/api/recommend/cold-start")
    print()
    
    uvicorn.run(
        "api.recommendation_api_with_coldstart:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
