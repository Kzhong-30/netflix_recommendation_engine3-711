import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from .content_recommender import ContentRecommender


class ColdStartRecommender:
    def __init__(self):
        self.content_recommender = ContentRecommender()
        self.movies_df = self.content_recommender.movies_df
        self.user_interaction_count = defaultdict(int)
        self.popular_genres = self._calculate_popular_genres()

    def _calculate_popular_genres(self) -> Dict[str, float]:
        genre_popularity = defaultdict(float)
        for _, movie in self.movies_df.iterrows():
            genres = movie['genres']
            if isinstance(genres, str):
                genres = eval(genres)
            for genre in genres:
                genre_popularity[genre] += movie['popularity'] * (movie['vote_average'] / 10.0)
        
        total = sum(genre_popularity.values())
        return {k: v / total for k, v in sorted(genre_popularity.items(), key=lambda x: x[1], reverse=True)}

    def _get_genre_match_score(self, movie_genres: List[str], preferred_genres: List[str]) -> float:
        if not preferred_genres:
            return 0.5
        
        match_count = len(set(preferred_genres) & set(movie_genres))
        return match_count / max(len(preferred_genres), len(movie_genres))

    def _get_diversity_penalty(self, selected_genres: List[str], movie_genres: List[str]) -> float:
        if not selected_genres:
            return 1.0
        
        genre_count = defaultdict(int)
        for genres in selected_genres:
            for g in genres:
                genre_count[g] += 1
        
        penalty = 1.0
        for g in movie_genres:
            if genre_count[g] >= 3:
                penalty *= 0.7
        
        return penalty

    def get_cold_start_recommendations(
        self,
        user_id: int,
        preferred_genres: List[str],
        num_recommendations: int = 10,
        enable_diversity: bool = True
    ) -> Dict[str, Any]:
        interaction_count = self.user_interaction_count[user_id]
        
        if interaction_count >= 10:
            strategy = "hybrid_mature"
        elif interaction_count >= 5:
            strategy = "hybrid_transition"
        else:
            strategy = "cold_start_genre_based"
        
        candidate_scores = []
        for idx, movie in self.movies_df.iterrows():
            movie_genres = movie['genres']
            if isinstance(movie_genres, str):
                movie_genres = eval(movie_genres)
            
            genre_score = self._get_genre_match_score(movie_genres, preferred_genres)
            quality_score = movie['vote_average'] / 10.0
            popularity_score = min(movie['popularity'] / 150.0, 1.0)
            
            if strategy == "cold_start_genre_based":
                final_score = 0.5 * genre_score + 0.3 * quality_score + 0.2 * popularity_score
            elif strategy == "hybrid_transition":
                final_score = 0.4 * genre_score + 0.4 * quality_score + 0.2 * popularity_score
            else:
                final_score = 0.3 * genre_score + 0.5 * quality_score + 0.2 * popularity_score
            
            candidate_scores.append({
                'idx': idx,
                'score': final_score,
                'genres': movie_genres,
                'genre_match': genre_score,
                'movie_data': movie
            })
        
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        
        if enable_diversity:
            recommendations = self._apply_diversity_selection(candidate_scores, num_recommendations)
        else:
            recommendations = candidate_scores[:num_recommendations]
        
        formatted_recommendations = []
        for rec in recommendations:
            movie = rec['movie_data']
            formatted_recommendations.append({
                'title': movie['title'],
                'genres': rec['genres'],
                'overview': movie['overview'],
                'vote_average': movie['vote_average'],
                'popularity': movie['popularity'],
                'match_score': float(rec['score']),
                'genre_match_score': float(rec['genre_match']),
                'recommendation_reason': self._get_recommendation_reason(rec, preferred_genres, strategy)
            })
        
        genre_diversity = self._calculate_genre_diversity(formatted_recommendations)
        
        return {
            'user_id': user_id,
            'strategy': strategy,
            'interaction_count': interaction_count,
            'recommendations': formatted_recommendations,
            'diversity_metrics': genre_diversity,
            'preferred_genres': preferred_genres
        }

    def _apply_diversity_selection(self, candidates: List[Dict], num_recommendations: int) -> List[Dict]:
        selected = []
        selected_genres = []
        used_indices = set()
        
        genre_coverage = set()
        for candidate in candidates:
            for g in candidate['genres']:
                genre_coverage.add(g)
            if len(genre_coverage) >= 8:
                break
        
        target_genres_per_slot = max(1, len(genre_coverage) // num_recommendations)
        
        while len(selected) < num_recommendations and len(candidates) > len(used_indices):
            best_score = -1
            best_candidate = None
            
            for candidate in candidates:
                if candidate['idx'] in used_indices:
                    continue
                
                diversity_penalty = self._get_diversity_penalty(selected_genres, candidate['genres'])
                adjusted_score = candidate['score'] * diversity_penalty
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                selected_genres.append(best_candidate['genres'])
                used_indices.add(best_candidate['idx'])
            else:
                break
        
        return selected

    def _calculate_genre_diversity(self, recommendations: List[Dict]) -> Dict[str, Any]:
        all_genres = []
        for rec in recommendations:
            all_genres.extend(rec['genres'])
        
        genre_counts = defaultdict(int)
        for g in all_genres:
            genre_counts[g] += 1
        
        total_genre_occurrences = len(all_genres)
        max_genre_percentage = max(genre_counts.values()) / total_genre_occurrences if total_genre_occurrences > 0 else 0
        
        unique_genres = len(genre_counts)
        
        return {
            'unique_genres': unique_genres,
            'max_genre_repetition': round(max_genre_percentage * 100, 2),
            'genre_distribution': dict(genre_counts),
            'diversity_pass': max_genre_percentage <= 0.30
        }

    def _get_recommendation_reason(self, rec: Dict, preferred_genres: List[str], strategy: str) -> str:
        matching_genres = list(set(preferred_genres) & set(rec['genres']))
        
        if matching_genres:
            return f"Based on your preference for {', '.join(matching_genres)} movies"
        elif rec['movie_data']['vote_average'] >= 8.0:
            return "Highly rated critically acclaimed movie"
        elif rec['movie_data']['popularity'] >= 100:
            return "Currently trending and popular movie"
        else:
            return "Carefully selected to maximize recommendation diversity"

    def increment_user_interaction(self, user_id: int):
        self.user_interaction_count[user_id] += 1

    def get_user_interaction_count(self, user_id: int) -> int:
        return self.user_interaction_count.get(user_id, 0)
