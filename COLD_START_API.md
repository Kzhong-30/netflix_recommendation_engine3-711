# 冷启动推荐 API 文档

## 端点: POST /api/recommend/cold-start

为新用户提供个性化的初始电影推荐。

### 功能特点

- 基于用户类型偏好的内容匹配
- TF-IDF 内容特征提取
- MMR（Maximal Marginal Relevance）多样性注入算法
- 响应时间 < 500ms

### 请求

**URL**: `POST /api/recommend/cold-start`

**Content-Type**: `application/json`

#### 请求体 (Request Body)

```json
{
    "user_id": 10001,
    "genres": ["Action", "Sci-Fi", "Thriller"],
    "num_recommendations": 10
}
```

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| user_id | integer | 是 | 新用户唯一标识 |
| genres | array[string] | 是 | 用户注册时选择的类型偏好，最少1个，最多5个 |
| num_recommendations | integer | 否 | 需要获取的推荐数量，范围1-20，默认10 |

### 响应

#### 成功响应 (200 OK)

```json
{
    "user_id": 10001,
    "recommendations": [
        {
            "movie": {
                "id": 0,
                "title": "The Dark Knight",
                "overview": "Batman faces the Joker in Gotham City",
                "genres": ["Action", "Crime", "Drama"],
                "vote_average": 9.0,
                "popularity": 100.0,
                "release_date": "2008-07-18"
            },
            "relevance_score": 0.85,
            "diversity_score": 0.12,
            "final_score": 0.82,
            "reason": "因为你喜欢Action类型"
        }
    ],
    "metadata": {
        "total_candidates": 15,
        "diversity_lambda": 0.5,
        "response_time_ms": 45.23,
        "preferred_genres": ["Action", "Sci-Fi", "Thriller"],
        "quality_metrics": {
            "accuracy": 0.8,
            "genre_diversity": 0.65,
            "avg_rating": 8.4
        }
    },
    "timestamp": "2026-04-20T10:30:00"
}
```

#### 响应字段说明

**recommendations** 数组中的每个对象包含：

| 字段 | 类型 | 描述 |
|------|------|------|
| movie | object | 电影详细信息 |
| relevance_score | float | 相关性分数 (0-1)，越高表示越符合用户偏好 |
| diversity_score | float | 多样性分数 |
| final_score | float | 最终综合分数 (MMR算法结果) |
| reason | string | 推荐理由说明 |

**movie** 对象包含：

| 字段 | 类型 | 描述 |
|------|------|------|
| id | integer | 电影ID |
| title | string | 电影标题 |
| overview | string | 电影简介 |
| genres | array[string] | 电影类型列表 |
| vote_average | float | 平均评分 (0-10) |
| popularity | float | 流行度分数 |
| release_date | string | 发行日期 (YYYY-MM-DD) |

### 错误响应

#### 400 Bad Request

```json
{
    "detail": "Invalid request: genres must contain 1-5 items"
}
```

#### 503 Service Unavailable

```json
{
    "detail": "Cold start recommendation system not available"
}
```

#### 500 Internal Server Error

```json
{
    "detail": "Failed to generate cold start recommendations: <error message>"
}
```

### 示例代码

#### Python

```python
import requests

url = "http://localhost:8000/api/recommend/cold-start"

payload = {
    "user_id": 10001,
    "genres": ["Action", "Sci-Fi", "Thriller"],
    "num_recommendations": 10
}

response = requests.post(url, json=payload)
recommendations = response.json()

for rec in recommendations["recommendations"]:
    print(f"{rec['movie']['title']} - {rec['reason']}")
```

#### cURL

```bash
curl -X POST "http://localhost:8000/api/recommend/cold-start" \
     -H "Content-Type: application/json" \
     -d '{
         "user_id": 10001,
         "genres": ["Action", "Sci-Fi", "Thriller"],
         "num_recommendations": 10
     }'
```

#### JavaScript (Fetch)

```javascript
const response = await fetch('http://localhost:8000/api/recommend/cold-start', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        user_id: 10001,
        genres: ['Action', 'Sci-Fi', 'Thriller'],
        num_recommendations: 10
    })
});

const data = await response.json();
console.log(data.recommendations);
```

## 其他端点

### GET /health

健康检查端点。

**响应示例**:
```json
{
    "status": "healthy",
    "version": "1.1.0",
    "cold_start_ready": true,
    "uptime_seconds": 3600
}
```

### GET /metrics

获取API性能指标。

**响应示例**:
```json
{
    "api_metrics": {
        "total_requests": 100,
        "successful_requests": 95,
        "failed_requests": 5,
        "avg_response_time": 0.045,
        "cold_start_requests": 30
    },
    "uptime_seconds": 3600,
    "cold_start_status": "ready"
}
```

## 验收标准

| 标准 | 目标 | 状态 |
|------|------|------|
| 冷启动推荐准确率 | > 40% | ✅ |
| 类型多样性 | 重复类型 ≤ 30% | ✅ |
| 响应时间 | < 500ms | ✅ |
| API文档 | 完整 | ✅ |
| Docker构建 | 成功 | ✅ |
| docker-compose | 包含Redis服务 | ✅ |

## 技术实现

### 核心组件

1. **ColdStartRecommender** (`app/cold_start.py`)
   - 冷启动推荐主类
   - 基于类型偏好的候选选择
   - 综合相关性评分

2. **TfidfFeatureExtractor** (`app/cold_start.py`)
   - TF-IDF内容特征提取
   - 使用 scikit-learn 实现
   - 支持电影间相似度计算

3. **inject_diversity** (`app/cold_start.py`)
   - MMR（Maximal Marginal Relevance）算法实现
   - 平衡相关性与多样性
   - 类型约束检查

### 算法流程

1. **候选选择**: 根据用户类型偏好选择高分电影
2. **特征提取**: 使用TF-IDF提取电影内容特征
3. **相关性评分**: 结合类型匹配、评分和流行度
4. **多样性注入**: 应用MMR算法重新排序
5. **类型约束**: 确保重复类型不超过30%
