# 🐳 Docker Deployment Guide

Complete guide for deploying the Netflix-Style Recommendation Engine with Docker.

## 🚀 Quick Deployment

### 1. Basic Single Container
```bash
# Build the image
docker build -t netflix-recommender .

# Run with environment variables
docker run -p 8000:8000 \
  -e TMDB_API_KEY=your_key_here \
  netflix-recommender
```

### 2. Production Stack with Docker Compose
```bash
# Set environment variables
cp .env.example .env
# Edit .env with your TMDB_API_KEY

# Start the full production stack
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- TMDB API Key
- 4GB+ RAM (for ML models)
- 10GB+ disk space

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Nginx       │    │   Recommendation  │    │   PostgreSQL    │
│  Load Balancer  │───▶│       API        │───▶│    Database     │
│   (Port 80/443) │    │    (Port 8000)   │    │   (Port 5432)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────────────┐               │
         │              │      Redis      │               │
         └──────────────│     Cache       │───────────────┘
                        │   (Port 6379)   │
                        └─────────────────┘
                                 │
         ┌─────────────────────────────────────────┐
         │              Monitoring                  │
         │  ┌─────────────┐    ┌─────────────┐    │
         │  │ Prometheus  │    │   Grafana   │    │
         │  │(Port 9090) │    │(Port 3000)  │    │
         │  └─────────────┘    └─────────────┘    │
         └─────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Configuration
TMDB_API_KEY=your_tmdb_api_key_here
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql://postgres:password@db:5432/recommendations
POSTGRES_DB=recommendations
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Security (production)
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# ML Model Configuration
N_FACTORS=50
MIN_SIMILARITY_THRESHOLD=0.3
MIN_RATING_THRESHOLD=6.0
```

## 🚀 Deployment Steps

### Step 1: Prepare Environment
```bash
# Clone repository
git clone https://github.com/yourusername/netflix-recommendation-engine.git
cd netflix-recommendation-engine

# Set up environment
cp .env.example .env
# Edit .env with your configuration
```

### Step 2: Build and Deploy
```bash
# Option A: Development deployment
docker-compose -f docker-compose.dev.yml up -d

# Option B: Production deployment
docker-compose -f docker-compose.yml up -d

# Option C: Scaled production deployment
docker-compose up -d --scale api=3
```

### Step 3: Verify Deployment
```bash
# Check all services are running
docker-compose ps

# Test API health
curl http://localhost/health

# Test recommendations
curl -X POST http://localhost/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "liked_movies": ["Superman"], "preferred_genres": ["Action"]}'

# Access monitoring dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## 📊 Monitoring & Observability

### Grafana Dashboards
- **API Performance**: Request rates, response times, error rates
- **ML Metrics**: Recommendation precision, recall, NDCG
- **System Resources**: CPU, memory, disk usage
- **Business Metrics**: User engagement, recommendation CTR

### Prometheus Metrics
```yaml
# Key metrics exposed by the API
- http_requests_total: Total HTTP requests
- http_request_duration_seconds: Request latency
- recommendation_precision_score: Model precision
- recommendation_recall_score: Model recall
- active_users_total: Number of active users
```

### Log Aggregation
```bash
# View API logs
docker-compose logs -f api

# View all service logs
docker-compose logs -f

# Filter logs by level
docker-compose logs -f api | grep ERROR
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificates for development
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/CN=localhost"

# For production, use Let's Encrypt or your certificate authority
```

### Database Security
```bash
# Create secure database password
openssl rand -base64 32

# Update docker-compose.yml with secure credentials
# Never use default passwords in production
```

### API Security
```python
# api/security.py - Add authentication middleware
from fastapi.security import HTTPBearer
from fastapi import Depends, HTTPException

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Implement JWT verification
    if not verify_jwt_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return token
```

## 📈 Scaling & Performance

### Horizontal Scaling
```bash
# Scale API instances
docker-compose up -d --scale api=5

# Scale with different resource limits
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

### Performance Tuning
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    environment:
      - WORKERS=4
      - MAX_CONNECTIONS=1000
```

### Caching Strategy
```python
# Add Redis caching to API
import redis
import json
from functools import wraps

redis_client = redis.from_url("redis://redis:6379/0")

def cache_recommendations(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"recommendations:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Get from ML model
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator
```

## 🧪 Testing in Docker

### Unit Tests
```bash
# Run tests in container
docker-compose exec api python -m pytest tests/ -v

# Run with coverage
docker-compose exec api python -m pytest tests/ --cov=models --cov-report=html
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load test against containerized API
locust -f tests/load_test.py --host http://localhost

# Or using Docker
docker run -p 8089:8089 -v $PWD:/mnt/locust locustio/locust \
  -f /mnt/locust/tests/load_test.py --host http://host.docker.internal
```

### Integration Tests
```bash
# Test full stack integration
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Test API endpoints
docker-compose exec api python tests/test_integration.py
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs api

# Common fixes:
# 1. Missing TMDB_API_KEY
# 2. Port already in use
# 3. Insufficient memory
```

#### Database Connection Issues
```bash
# Check database status
docker-compose exec db pg_isready -U postgres

# Reset database
docker-compose down -v
docker-compose up -d db
```

#### Memory Issues
```bash
# Monitor resource usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  api:
    mem_limit: 4g
```

#### SSL Certificate Issues
```bash
# Regenerate certificates
rm -rf nginx/ssl/*
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem
```

### Performance Debugging
```bash
# Check API response times
docker-compose exec api python -c "
import requests
import time
start = time.time()
r = requests.get('http://localhost:8000/health')
print(f'Response time: {time.time() - start:.2f}s')
print(f'Status: {r.status_code}')
"

# Monitor database performance
docker-compose exec db psql -U postgres -d recommendations -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/docker-deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build and push Docker image
        run: |
          docker build -t netflix-recommender:${{ github.sha }} .
          docker tag netflix-recommender:${{ github.sha }} netflix-recommender:latest
          
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.prod.yml up -d
          
      - name: Health check
        run: |
          sleep 30
          curl -f http://localhost/health || exit 1
```

## 📦 Production Checklist

### Before Deployment
- [ ] Set secure passwords for all services
- [ ] Configure SSL certificates
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Set resource limits and requests
- [ ] Enable health checks
- [ ] Configure backups

### After Deployment
- [ ] Verify all services are healthy
- [ ] Test API endpoints
- [ ] Check monitoring dashboards
- [ ] Verify SSL certificates
- [ ] Test failover scenarios
- [ ] Document runbooks
- [ ] Set up alerts

## 🔧 Maintenance

### Regular Tasks
```bash
# Update dependencies
docker-compose pull
docker-compose up -d

# Backup database
docker-compose exec db pg_dump -U postgres recommendations > backup.sql

# Clean up unused containers and images
docker system prune -a

# Rotate logs
docker-compose exec nginx logrotate /etc/logrotate.d/nginx
```

### Monitoring Health
```bash
# Check all services
docker-compose ps

# Monitor resource usage
docker stats

# Check logs for errors
docker-compose logs --tail=100 api | grep ERROR
```

## 📞 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review container logs: `docker-compose logs [service]`
3. Verify environment configuration
4. Check resource constraints
5. Open an issue on GitHub

---

**🎉 Congratulations! You now have a production-ready, containerized Netflix-style recommendation engine!**