"""
冷启动用户处理模块 - Cold Start Recommendation System

该模块解决新用户推荐质量差的问题，通过以下策略：
1. 基于类型偏好的内容推荐
2. 热门和趋势内容混合
3. TF-IDF内容特征提取
4. MMR（Maximal Marginal Relevance）多样性注入
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import time


@dataclass
class MovieItem:
    """电影数据项"""
    title: str
    overview: str
    genres: List[str]
    vote_average: float
    popularity: float
    release_date: str
    id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "overview": self.overview,
            "genres": self.genres,
            "vote_average": self.vote_average,
            "popularity": self.popularity,
            "release_date": self.release_date
        }


@dataclass
class RecommendationResult:
    """推荐结果"""
    movie: MovieItem
    relevance_score: float
    diversity_score: float
    final_score: float
    reason: str


class TfidfFeatureExtractor:
    """基于TF-IDF的内容特征提取器"""
    
    def __init__(self, max_features: int = 5000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            dtype=np.float32
        )
        self.feature_matrix = None
        self.movies = []
        self.is_fitted = False
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本：清理和标准化"""
        if not isinstance(text, str):
            return ""
        # 转小写
        text = text.lower()
        # 移除特殊字符，保留字母和数字
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def fit_transform(self, movies: List[MovieItem]) -> np.ndarray:
        """训练并转换电影数据"""
        self.movies = movies
        
        # 组合文本特征：概述 + 类型
        texts = []
        for movie in movies:
            overview = self.preprocess_text(movie.overview)
            genres_text = ' '.join(movie.genres) if movie.genres else ''
            combined_text = f"{overview} {genres_text}"
            texts.append(combined_text)
        
        # 训练TF-IDF并转换
        self.feature_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        
        return self.feature_matrix
    
    def transform(self, text: str) -> np.ndarray:
        """转换单个文本"""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        processed_text = self.preprocess_text(text)
        return self.vectorizer.transform([processed_text])
    
    def get_similarity_matrix(self) -> np.ndarray:
        """获取电影间相似度矩阵"""
        if self.feature_matrix is None:
            raise ValueError("Must call fit_transform first")
        return cosine_similarity(self.feature_matrix)
    
    def find_similar_movies(self, query_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """查找相似电影"""
        similarity_matrix = self.get_similarity_matrix()
        similarities = similarity_matrix[query_idx]
        # 排除自身
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        return [(idx, similarities[idx]) for idx in similar_indices]


def inject_diversity(
    candidates: List[RecommendationResult],
    similarity_matrix: np.ndarray,
    lambda_param: float = 0.5,
    max_genre_ratio: float = 0.3
) -> List[RecommendationResult]:
    """
    使用MMR（Maximal Marginal Relevance）算法注入多样性
    
    Args:
        candidates: 候选推荐列表
        similarity_matrix: 电影间相似度矩阵
        lambda_param: 相关性与多样性的权衡参数 (0-1)
                       0 = 纯多样性, 1 = 纯相关性
        max_genre_ratio: 单个类型最大占比
    
    Returns:
        重新排序后的推荐列表
    """
    if not candidates:
        return []
    
    selected = []
    remaining = list(range(len(candidates)))
    
    # 首先选择相关性最高的
    first_idx = max(remaining, key=lambda i: candidates[i].relevance_score)
    selected.append(first_idx)
    remaining.remove(first_idx)
    
    # MMR选择
    while remaining and len(selected) < len(candidates):
        mmr_scores = []
        
        for idx in remaining:
            relevance = candidates[idx].relevance_score
            
            # 计算与已选项目的最大相似度
            max_sim = 0.0
            for sel_idx in selected:
                sim = similarity_matrix[idx][sel_idx]
                max_sim = max(max_sim, sim)
            
            # MMR分数 = λ * Relevance - (1-λ) * max_sim
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((idx, mmr_score))
        
        # 选择MMR分数最高的
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    # 检查类型多样性约束
    result = [candidates[i] for i in selected]
    result = _enforce_genre_diversity(result, max_genre_ratio)
    
    return result


def _enforce_genre_diversity(
    recommendations: List[RecommendationResult],
    max_genre_ratio: float
) -> List[RecommendationResult]:
    """
    强制执行类型多样性约束
    
    确保推荐列表中重复类型不超过指定比例
    """
    if not recommendations:
        return []
    
    max_count = int(len(recommendations) * max_genre_ratio)
    if max_count < 1:
        max_count = 1
    
    # 统计类型出现次数
    genre_counts = Counter()
    for rec in recommendations:
        for genre in rec.movie.genres:
            genre_counts[genre] += 1
    
    # 如果所有类型都符合要求，直接返回
    if all(count <= max_count for count in genre_counts.values()):
        return recommendations
    
    # 否则重新排序以平衡类型
    balanced = []
    genre_counts = Counter()
    
    # 按最终分数排序
    sorted_recs = sorted(recommendations, key=lambda x: x.final_score, reverse=True)
    
    for rec in sorted_recs:
        # 检查该电影的类型是否都未超限
        can_add = True
        for genre in rec.movie.genres:
            if genre_counts[genre] >= max_count:
                can_add = False
                break
        
        if can_add:
            balanced.append(rec)
            for genre in rec.movie.genres:
                genre_counts[genre] += 1
    
    # 如果筛选后数量太少，补充剩余的高分推荐
    if len(balanced) < len(recommendations) * 0.7:
        existing_titles = {r.movie.title for r in balanced}
        for rec in sorted_recs:
            if rec.movie.title not in existing_titles:
                balanced.append(rec)
                if len(balanced) >= len(recommendations):
                    break
    
    return balanced if balanced else recommendations


class ColdStartRecommender:
    """
    冷启动推荐器 - 为新用户提供高质量的初始推荐
    
    特点：
    1. 基于类型偏好的内容匹配
    2. 结合热门度和评分
    3. TF-IDF内容特征提取
    4. MMR多样性注入
    """
    
    def __init__(self, movies_df: Optional[pd.DataFrame] = None):
        self.movies_df = movies_df
        self.feature_extractor = TfidfFeatureExtractor()
        self.movies: List[MovieItem] = []
        self.genre_popularity: Dict[str, List[MovieItem]] = {}
        self.is_trained = False
        
    def load_movies_data(self, movies_df: pd.DataFrame):
        """加载电影数据"""
        self.movies_df = movies_df
        self._prepare_movies()
        
    def _prepare_movies(self):
        """准备电影数据"""
        if self.movies_df is None:
            raise ValueError("Movies data not loaded")
        
        self.movies = []
        self.genre_popularity = {}
        
        for idx, row in self.movies_df.iterrows():
            # 解析类型
            genres = row.get('genres', [])
            if isinstance(genres, str):
                try:
                    import ast
                    genres = ast.literal_eval(genres)
                except:
                    genres = [g.strip() for g in genres.split(',')]
            
            movie = MovieItem(
                id=idx,
                title=str(row.get('title', '')),
                overview=str(row.get('overview', '')),
                genres=genres if isinstance(genres, list) else [],
                vote_average=float(row.get('vote_average', 0)),
                popularity=float(row.get('popularity', 0)),
                release_date=str(row.get('release_date', ''))
            )
            
            self.movies.append(movie)
            
            # 按类型分组
            for genre in movie.genres:
                if genre not in self.genre_popularity:
                    self.genre_popularity[genre] = []
                self.genre_popularity[genre].append(movie)
        
        # 训练TF-IDF
        if self.movies:
            self.feature_extractor.fit_transform(self.movies)
            self.is_trained = True
    
    def get_cold_start_recommendations(
        self,
        preferred_genres: List[str],
        num_recommendations: int = 10,
        diversity_lambda: float = 0.5
    ) -> Dict[str, Any]:
        """
        获取冷启动推荐
        
        Args:
            preferred_genres: 用户偏好的类型列表
            num_recommendations: 推荐数量
            diversity_lambda: 多样性参数
        
        Returns:
            包含推荐结果的字典
        """
        start_time = time.time()
        
        if not self.is_trained:
            return {
                "success": False,
                "error": "Recommender not trained",
                "recommendations": []
            }
        
        # 1. 基于类型偏好获取候选集
        candidates = self._get_candidates_by_genres(preferred_genres)
        
        if not candidates:
            # 如果没有类型匹配，使用全局热门
            candidates = self._get_global_popular(num_recommendations * 3)
        
        # 2. 计算相关性分数
        candidates_with_scores = self._compute_relevance_scores(
            candidates, preferred_genres
        )
        
        # 3. 应用MMR多样性注入
        similarity_matrix = self.feature_extractor.get_similarity_matrix()
        diversified = inject_diversity(
            candidates_with_scores,
            similarity_matrix,
            lambda_param=diversity_lambda,
            max_genre_ratio=0.3
        )
        
        # 4. 限制数量并格式化结果
        final_recommendations = diversified[:num_recommendations]
        
        response_time = time.time() - start_time
        
        return {
            "success": True,
            "recommendations": [
                {
                    "movie": rec.movie.to_dict(),
                    "relevance_score": round(rec.relevance_score, 3),
                    "diversity_score": round(rec.diversity_score, 3),
                    "final_score": round(rec.final_score, 3),
                    "reason": rec.reason
                }
                for rec in final_recommendations
            ],
            "metadata": {
                "total_candidates": len(candidates),
                "diversity_lambda": diversity_lambda,
                "response_time_ms": round(response_time * 1000, 2),
                "preferred_genres": preferred_genres
            }
        }
    
    def _get_candidates_by_genres(
        self,
        preferred_genres: List[str],
        min_rating: float = 6.0
    ) -> List[MovieItem]:
        """基于类型偏好获取候选电影"""
        candidates = []
        seen_titles = set()
        
        for genre in preferred_genres:
            if genre in self.genre_popularity:
                # 获取该类型的高分电影
                genre_movies = [
                    m for m in self.genre_popularity[genre]
                    if m.vote_average >= min_rating and m.title not in seen_titles
                ]
                
                # 按综合分数排序 (评分 * 0.7 + 流行度归一化 * 0.3)
                max_pop = max(m.popularity for m in genre_movies) if genre_movies else 1
                genre_movies.sort(
                    key=lambda m: m.vote_average * 0.7 + (m.popularity / max_pop) * 30 * 0.3,
                    reverse=True
                )
                
                # 每个类型取前15部
                for movie in genre_movies[:15]:
                    candidates.append(movie)
                    seen_titles.add(movie.title)
        
        return candidates
    
    def _get_global_popular(self, num_movies: int) -> List[MovieItem]:
        """获取全局热门电影"""
        sorted_movies = sorted(
            self.movies,
            key=lambda m: m.vote_average * 0.6 + min(m.popularity / 100, 10) * 0.4,
            reverse=True
        )
        return sorted_movies[:num_movies]
    
    def _compute_relevance_scores(
        self,
        candidates: List[MovieItem],
        preferred_genres: List[str]
    ) -> List[RecommendationResult]:
        """计算候选电影的相关性分数"""
        results = []
        
        # 归一化流行度
        max_popularity = max(m.popularity for m in candidates) if candidates else 1
        
        for movie in candidates:
            # 类型匹配分数
            genre_match_score = self._calculate_genre_match(movie, preferred_genres)
            
            # 评分分数 (归一化到0-1)
            rating_score = movie.vote_average / 10.0
            
            # 流行度分数
            popularity_score = min(movie.popularity / max_popularity, 1.0)
            
            # 综合相关性分数
            relevance = (
                genre_match_score * 0.5 +
                rating_score * 0.35 +
                popularity_score * 0.15
            )
            
            # 生成推荐理由
            reason = self._generate_reason(movie, preferred_genres, genre_match_score)
            
            result = RecommendationResult(
                movie=movie,
                relevance_score=relevance,
                diversity_score=0.0,  # 将在MMR中计算
                final_score=relevance,
                reason=reason
            )
            results.append(result)
        
        return results
    
    def _calculate_genre_match(
        self,
        movie: MovieItem,
        preferred_genres: List[str]
    ) -> float:
        """计算类型匹配分数"""
        if not preferred_genres or not movie.genres:
            return 0.0
        
        movie_genres_set = set(g.lower() for g in movie.genres)
        preferred_set = set(g.lower() for g in preferred_genres)
        
        matches = len(movie_genres_set & preferred_set)
        total_unique = len(movie_genres_set | preferred_set)
        
        if total_unique == 0:
            return 0.0
        
        # Jaccard相似度
        return matches / total_unique
    
    def _generate_reason(
        self,
        movie: MovieItem,
        preferred_genres: List[str],
        match_score: float
    ) -> str:
        """生成推荐理由"""
        if match_score > 0.7:
            matched_genres = [g for g in movie.genres if g in preferred_genres]
            return f"因为你喜欢{', '.join(matched_genres[:2])}类型"
        elif movie.vote_average >= 8.0:
            return "高分好评电影"
        elif movie.popularity > 100:
            return "当下热门"
        else:
            return "可能符合你的口味"
    
    def evaluate_recommendation_quality(
        self,
        recommendations: List[Dict[str, Any]],
        preferred_genres: List[str]
    ) -> Dict[str, float]:
        """
        评估推荐质量指标
        
        用于验收标准验证
        """
        if not recommendations:
            return {"accuracy": 0.0, "genre_diversity": 0.0}
        
        # 1. 准确率：推荐中匹配类型的比例
        total_matches = 0
        for rec in recommendations:
            movie_genres = set(g.lower() for g in rec["movie"]["genres"])
            preferred_set = set(g.lower() for g in preferred_genres)
            if movie_genres & preferred_set:
                total_matches += 1
        
        accuracy = total_matches / len(recommendations)
        
        # 2. 类型多样性
        all_genres = []
        for rec in recommendations:
            all_genres.extend(rec["movie"]["genres"])
        
        genre_counts = Counter(all_genres)
        if genre_counts:
            max_genre_count = max(genre_counts.values())
            total_genre_instances = len(all_genres)
            diversity = 1 - (max_genre_count / total_genre_instances)
        else:
            diversity = 0.0
        
        return {
            "accuracy": round(accuracy, 3),
            "genre_diversity": round(diversity, 3),
            "avg_rating": round(
                sum(r["movie"]["vote_average"] for r in recommendations) / len(recommendations), 2
            )
        }


# 全局冷启动推荐器实例
cold_start_recommender: Optional[ColdStartRecommender] = None


def get_cold_start_recommender() -> Optional[ColdStartRecommender]:
    """获取全局冷启动推荐器实例"""
    global cold_start_recommender
    return cold_start_recommender


def initialize_cold_start_recommender(movies_df: pd.DataFrame) -> ColdStartRecommender:
    """初始化冷启动推荐器"""
    global cold_start_recommender
    cold_start_recommender = ColdStartRecommender(movies_df)
    return cold_start_recommender
