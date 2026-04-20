# Cold-Start Recommendation API Documentation

## Overview

The Cold-Start Recommendation API provides personalized movie recommendations for new users who have no viewing history. It uses TF-IDF based content features and MMR (Maximal Marginal Relevance) algorithm for diversity injection.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check the API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_status": "ready",
  "uptime_seconds": 123.45,
  "movies_loaded": 80
}
```

---

### 2. Get Available Genres

**GET** `/api/genres`

Retrieve all available genres in the system.

**Response:**
```json
{
  "genres": [
    {
      "name": "Action",
      "movie_count": 25,
      "avg_rating": 7.5
    },
    {
      "name": "Adventure",
      "movie_count": 20,
      "avg_rating": 7.3
    }
  ],
  "total_count": 19
}
```

---

### 3. Cold-Start Recommendations

**POST** `/api/recommend/cold-start`

Get personalized recommendations for a new user based on their genre preferences.

#### Request Body

```json
{
  "user_id": 123,
  "preferred_genres": ["Action", "Adventure", "Sci-Fi"],
  "num_recommendations": 10,
  "min_rating": 6.0,
  "diversity_lambda": 0.7
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `user_id` | integer | No | null | Optional user identifier |
| `preferred_genres` | array[string] | Yes | - | List of user's preferred genres |
| `num_recommendations` | integer | No | 10 | Number of recommendations (1-50) |
| `min_rating` | float | No | 6.0 | Minimum movie rating threshold (0-10) |
| `diversity_lambda` | float | No | 0.7 | Balance between relevance and diversity (0-1) |

#### Response

```json
{
  "user_id": 123,
  "strategy": "cold_start_genre_based",
  "recommendations": [
    {
      "title": "The Dark Knight",
      "genres": ["Action", "Crime", "Drama"],
      "vote_average": 9.0,
      "overview": "When the menace known as the Joker...",
      "release_date": "2008-07-18",
      "relevance_score": 0.85,
      "genre_match_count": 1,
      "mmr_applied": true
    }
  ],
  "metadata": {
    "strategy": "cold_start_genre_based",
    "preferred_genres": ["Action", "Adventure", "Sci-Fi"],
    "candidate_pool_size": 45,
    "inference_time_ms": 125.5,
    "diversity_applied": true,
    "diversity_lambda": 0.7
  },
  "diversity_metrics": {
    "genre_diversity": 0.8,
    "unique_genres": 8,
    "genre_repetition": 0.25,
    "total_genres": 20,
    "genre_distribution": {
      "Action": 5,
      "Adventure": 4,
      "Sci-Fi": 3,
      "Drama": 3,
      "Thriller": 2,
      "Comedy": 2,
      "Horror": 1
    }
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | string | Recommendation strategy used |
| `recommendations` | array | List of recommended movies |
| `metadata` | object | Additional information about the recommendation process |
| `diversity_metrics` | object | Metrics about genre diversity in recommendations |

---

## Performance Guarantees

| Metric | Target | Description |
|--------|--------|-------------|
| Response Time | < 500ms | API response time for recommendations |
| Accuracy | > 40% | Relevance of first 10 recommendations |
| Diversity | < 30% | Maximum genre repetition ratio |

---

## Example Usage

### cURL

```bash
curl -X POST "http://localhost:8000/api/recommend/cold-start" \
  -H "Content-Type: application/json" \
  -d '{
    "preferred_genres": ["Action", "Adventure"],
    "num_recommendations": 10
  }'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/recommend/cold-start",
    json={
        "preferred_genres": ["Action", "Adventure", "Sci-Fi"],
        "num_recommendations": 10,
        "min_rating": 6.5,
        "diversity_lambda": 0.7
    }
)

data = response.json()
for movie in data["recommendations"]:
    print(f"{movie['title']} - Rating: {movie['vote_average']}")
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/recommend/cold-start', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    preferred_genres: ['Action', 'Adventure'],
    num_recommendations: 10
  })
});

const data = await response.json();
console.log(data.recommendations);
```

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Invalid request parameters"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Failed to generate recommendations: error message"
}
```

### 503 Service Unavailable

```json
{
  "detail": "Recommender system not available"
}
```

---

## Algorithm Details

### TF-IDF Content Features

The system uses scikit-learn's TfidfVectorizer to extract content features from:
- Movie overview text
- Genre information
- Movie title
- Tagline (if available)

### MMR (Maximal Marginal Relevance)

Diversity is injected using the MMR algorithm:

```
MMR = λ * Sim(item, query) - (1-λ) * max(Sim(item, selected))
```

Where:
- `λ` (lambda) controls the balance between relevance and diversity
- Higher λ values prioritize relevance
- Lower λ values prioritize diversity

### Genre Constraint

The system ensures that no single genre appears in more than 30% of the recommendation slots, providing diverse content discovery.
