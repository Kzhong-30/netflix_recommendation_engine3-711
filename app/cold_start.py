import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import time
from .recommenders import ContentRecommender, ColdStartRecommender


def inject_diversity(
    recommendations: List[Dict[str, Any]],
    lambda_param: float = 0.5,
    top_n: int = 10,
    content_recommender: ContentRecommender = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Maximal Marginal Relevance (MMR) algorithm for diversity injection
    
    Args:
        recommendations: List of scored recommendation candidates
        lambda_param: Trade-off between relevance and diversity (0=diversity only, 1=relevance only)
        top_n: Number of final recommendations to return
        content_recommender: Content recommender instance for feature vectors
    
    Returns:
        Tuple of (diversified recommendations, diversity metrics)
    """
    if len(recommendations) <= top_n:
        return recommendations, _calculate_diversity_metrics(recommendations)
    
    if not content_recommender:
        content_recommender = ContentRecommender()
    
    movies_df = content_recommender.movies_df
    tfidf_matrix = content_recommender.tfidf_matrix
    
    title_to_idx = {movie['title']: i for i, movie in movies_df.iterrows()}
    
    candidate_indices = []
    candidate_scores = []
    valid_recommendations = []
    
    for rec in recommendations:
        title = rec['title']
        if title in title_to_idx:
            candidate_indices.append(title_to_idx[title])
            candidate_scores.append(rec.get('match_score', 0.5))
            valid_recommendations.append(rec)
    
    if len(candidate_indices) < top_n:
        return recommendations[:top_n], _calculate_diversity_metrics(recommendations[:top_n])
    
    candidate_features = tfidf_matrix[candidate_indices].toarray()
    similarity_matrix = cosine_similarity(candidate_features)
    
    selected = []
    remaining = list(range(len(candidate_indices)))
    
    first_idx = np.argmax(candidate_scores)
    selected.append(first_idx)
    remaining.remove(first_idx)
    
    while len(selected) < top_n and remaining:
        mmr_scores = []
        for i in remaining:
            relevance = candidate_scores[i]
            
            max_similarity = max([similarity_matrix[i][s] for s in selected]) if selected else 0
            
            mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((i, mmr))
        
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    diversified_recommendations = [valid_recommendations[i] for i in selected]
    diversity_metrics = _calculate_diversity_metrics(diversified_recommendations)
    
    return diversified_recommendations, diversity_metrics


def _calculate_diversity_metrics(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate diversity metrics for recommendations"""
    all_genres = []
    genre_sets = []
    
    for rec in recommendations:
        genres = rec.get('genres', [])
        if isinstance(genres, str):
            genres = eval(genres)
        all_genres.extend(genres)
        genre_sets.append(set(genres))
    
    genre_counts = defaultdict(int)
    for g in all_genres:
        genre_counts[g] += 1
    
    total_genre_occurrences = len(all_genres)
    max_genre_percentage = max(genre_counts.values()) / total_genre_occurrences if total_genre_occurrences > 0 else 0
    
    avg_pairwise_similarity = 0.0
    pair_count = 0
    for i in range(len(genre_sets)):
        for j in range(i + 1, len(genre_sets)):
            intersection = len(genre_sets[i] & genre_sets[j])
            union = len(genre_sets[i] | genre_sets[j])
            if union > 0:
                avg_pairwise_similarity += intersection / union
                pair_count += 1
    avg_pairwise_similarity = avg_pairwise_similarity / pair_count if pair_count > 0 else 0
    
    return {
        'unique_genres': len(genre_counts),
        'max_genre_repetition_percent': round(max_genre_percentage * 100, 2),
        'avg_genre_similarity': round(avg_pairwise_similarity, 3),
        'diversity_constraint_met': max_genre_percentage <= 0.30,
        'genre_distribution': dict(genre_counts)
    }


class ColdStartHandler:
    def __init__(self):
        self.cold_start_recommender = ColdStartRecommender()
        self.content_recommender = ContentRecommender()
        self.user_transition_threshold = 10
        self._warm_up()
    
    def _warm_up(self):
        """Warm up the model to ensure fast first response"""
        start_time = time.time()
        _ = self.get_cold_start_recommendations(
            user_id=-1,
            preferred_genres=["Action", "Drama"],
            num_recommendations=10
        )
        self.warmup_time = time.time() - start_time
    
    def get_cold_start_recommendations(
        self,
        user_id: int,
        preferred_genres: List[str],
        num_recommendations: int = 10,
        lambda_param: float = 0.6
    ) -> Dict[str, Any]:
        """
        Get cold start recommendations for a new user with MMR diversity injection
        
        Args:
            user_id: User identifier
            preferred_genres: List of user's preferred genres
            num_recommendations: Number of recommendations to return
            lambda_param: MMR lambda parameter (relevance vs diversity trade-off)
        
        Returns:
            Cold start recommendation response with diversity metrics
        """
        start_time = time.time()
        
        base_recommendations = self.cold_start_recommender.get_cold_start_recommendations(
            user_id=user_id,
            preferred_genres=preferred_genres,
            num_recommendations=min(num_recommendations * 3, 30),
            enable_diversity=False
        )
        
        diversified_recs, diversity_metrics = inject_diversity(
            recommendations=base_recommendations['recommendations'],
            lambda_param=lambda_param,
            top_n=num_recommendations,
            content_recommender=self.content_recommender
        )
        
        accuracy_estimate = self._estimate_accuracy(diversified_recs, preferred_genres)
        
        response_time = time.time() - start_time
        
        return {
            'user_id': user_id,
            'strategy': base_recommendations['strategy'],
            'interaction_count': base_recommendations['interaction_count'],
            'is_new_user': base_recommendations['interaction_count'] < self.user_transition_threshold,
            'preferred_genres': preferred_genres,
            'recommendations': diversified_recs,
            'diversity_metrics': diversity_metrics,
            'performance_metrics': {
                'accuracy_estimate_percent': round(accuracy_estimate * 100, 2),
                'response_time_ms': round(response_time * 1000, 2),
                'warmup_time_ms': round(self.warmup_time * 1000, 2),
                'response_time_constraint_met': response_time < 0.5
            },
            'accuracy_constraint_met': accuracy_estimate >= 0.40
        }
    
    def _estimate_accuracy(self, recommendations: List[Dict[str, Any]], preferred_genres: List[str]) -> float:
        """Estimate recommendation accuracy based on genre match and quality scores"""
        if not preferred_genres:
            return 0.5
        
        matches = 0
        for rec in recommendations:
            genres = rec.get('genres', [])
            genre_match = len(set(preferred_genres) & set(genres)) > 0
            high_quality = rec.get('vote_average', 0) >= 7.0
            
            if genre_match or high_quality:
                matches += 1
        
        return matches / len(recommendations)
    
    def record_user_interaction(self, user_id: int):
        """Record a user interaction to transition out of cold start"""
        self.cold_start_recommender.increment_user_interaction(user_id)
    
    def get_user_status(self, user_id: int) -> Dict[str, Any]:
        """Get user cold start status"""
        interaction_count = self.cold_start_recommender.get_user_interaction_count(user_id)
        is_in_cold_start = interaction_count < self.user_transition_threshold
        
        return {
            'user_id': user_id,
            'interaction_count': interaction_count,
            'is_in_cold_start': is_in_cold_start,
            'interactions_to_exit': max(0, self.user_transition_threshold - interaction_count)
        }


cold_start_handler = ColdStartHandler()
