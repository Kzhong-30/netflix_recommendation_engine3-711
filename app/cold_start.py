import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


class ColdStartRecommender:
    """
    Cold start recommender for new users without viewing history.
    Uses TF-IDF based content features and genre preferences to generate
    initial recommendations.
    """
    
    def __init__(self, movies_df: Optional[pd.DataFrame] = None):
        self.movies_df = movies_df
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.genre_vectors = {}
        self.is_fitted = False
        self._fit_time = 0
        
    def fit(self, movies_df: pd.DataFrame) -> 'ColdStartRecommender':
        """
        Fit the cold start recommender with movie data.
        
        Args:
            movies_df: DataFrame with columns: title, genres, overview, vote_average, etc.
        """
        start_time = time.time()
        self.movies_df = movies_df.copy()
        
        self._prepare_content_features()
        self._build_genre_vectors()
        
        self._fit_time = time.time() - start_time
        self.is_fitted = True
        return self
    
    def _prepare_content_features(self):
        """Prepare TF-IDF features from movie content."""
        self.movies_df['content_features'] = self._build_content_text()
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['content_features'].fillna('')
        )
    
    def _build_content_text(self) -> pd.Series:
        """Build combined text for TF-IDF from movie attributes."""
        content_parts = []
        
        if 'overview' in self.movies_df.columns:
            content_parts.append(self.movies_df['overview'].fillna('').astype(str))
        
        if 'genres' in self.movies_df.columns:
            genres_text = self.movies_df['genres'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x).replace(',', ' ').replace('[', '').replace(']', '').replace("'", '')
            )
            content_parts.append(genres_text)
        
        if 'title' in self.movies_df.columns:
            content_parts.append(self.movies_df['title'].fillna('').astype(str))
        
        if 'tagline' in self.movies_df.columns:
            content_parts.append(self.movies_df['tagline'].fillna('').astype(str))
        
        combined = content_parts[0]
        for part in content_parts[1:]:
            combined = combined + ' ' + part
        
        return combined
    
    def _build_genre_vectors(self):
        """Build genre-specific feature vectors for quick lookup."""
        unique_genres = set()
        
        for genres in self.movies_df['genres']:
            if isinstance(genres, list):
                unique_genres.update(genres)
            elif isinstance(genres, str):
                cleaned = genres.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
                unique_genres.update([g.strip() for g in cleaned.split(',') if g.strip()])
        
        for genre in unique_genres:
            genre_mask = self.movies_df['genres'].apply(
                lambda x: genre in x if isinstance(x, list) else genre in str(x)
            )
            
            if genre_mask.sum() > 0:
                genre_indices = genre_mask[genre_mask].index.tolist()
                self.genre_vectors[genre] = {
                    'indices': genre_indices,
                    'movies': self.movies_df[genre_mask].copy()
                }
    
    def recommend(
        self,
        preferred_genres: List[str],
        num_recommendations: int = 10,
        min_rating: float = 6.0,
        diversity_lambda: float = 0.7
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate recommendations for a new user based on genre preferences.
        
        Args:
            preferred_genres: List of genres the user prefers
            num_recommendations: Number of recommendations to return
            min_rating: Minimum movie rating threshold
            diversity_lambda: Lambda parameter for MMR diversity (0-1)
            
        Returns:
            Tuple of (recommendations list, metadata dict)
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be fitted before making recommendations")
        
        start_time = time.time()
        
        candidate_movies = self._get_candidate_movies(preferred_genres, min_rating)
        
        if len(candidate_movies) == 0:
            candidate_movies = self._get_fallback_movies(min_rating)
        
        scored_movies = self._score_by_genre_relevance(candidate_movies, preferred_genres)
        
        from app.recommenders.diversity import inject_diversity
        diverse_recommendations = inject_diversity(
            candidates=scored_movies,
            tfidf_matrix=self.tfidf_matrix,
            movies_df=self.movies_df,
            num_recommendations=num_recommendations,
            lambda_param=diversity_lambda
        )
        
        inference_time = time.time() - start_time
        
        metadata = {
            'strategy': 'cold_start_genre_based',
            'preferred_genres': preferred_genres,
            'candidate_pool_size': len(candidate_movies),
            'inference_time_ms': round(inference_time * 1000, 2),
            'diversity_applied': True,
            'diversity_lambda': diversity_lambda
        }
        
        return diverse_recommendations, metadata
    
    def _get_candidate_movies(
        self,
        preferred_genres: List[str],
        min_rating: float
    ) -> pd.DataFrame:
        """Get candidate movies matching user's preferred genres."""
        genre_mask = self.movies_df['genres'].apply(
            lambda x: any(
                genre in (x if isinstance(x, list) else str(x).replace('[', '').replace(']', '').replace("'", '').split(','))
                for genre in preferred_genres
            ) if pd.notna(x) else False
        )
        
        rating_mask = self.movies_df['vote_average'] >= min_rating
        
        candidates = self.movies_df[genre_mask & rating_mask].copy()
        
        if len(candidates) > 0:
            candidates = candidates.sort_values('vote_average', ascending=False)
        
        return candidates
    
    def _get_fallback_movies(self, min_rating: float) -> pd.DataFrame:
        """Get fallback popular movies when no genre matches."""
        rating_mask = self.movies_df['vote_average'] >= min_rating
        candidates = self.movies_df[rating_mask].copy()
        candidates = candidates.sort_values('vote_average', ascending=False)
        return candidates.head(100)
    
    def _score_by_genre_relevance(
        self,
        candidates: pd.DataFrame,
        preferred_genres: List[str]
    ) -> List[Dict[str, Any]]:
        """Score movies by genre relevance and quality."""
        scored = []
        
        for idx, movie in candidates.iterrows():
            movie_genres = movie['genres']
            if isinstance(movie_genres, str):
                movie_genres = [g.strip() for g in movie_genres.replace('[', '').replace(']', '').replace("'", '').split(',')]
            
            genre_match_count = sum(1 for g in movie_genres if g in preferred_genres)
            genre_relevance = genre_match_count / len(preferred_genres) if preferred_genres else 0
            
            quality_score = movie['vote_average'] / 10.0
            
            final_score = 0.6 * genre_relevance + 0.4 * quality_score
            
            scored.append({
                'index': idx,
                'title': movie.get('title', ''),
                'genres': movie_genres,
                'vote_average': movie.get('vote_average', 0),
                'overview': movie.get('overview', ''),
                'release_date': movie.get('release_date', ''),
                'score': final_score,
                'genre_match_count': genre_match_count
            })
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored
    
    def get_genre_statistics(self) -> Dict[str, Any]:
        """Get statistics about available genres."""
        if not self.is_fitted:
            return {}
        
        stats = {}
        for genre, data in self.genre_vectors.items():
            stats[genre] = {
                'movie_count': len(data['indices']),
                'avg_rating': data['movies']['vote_average'].mean() if 'vote_average' in data['movies'].columns else 0
            }
        
        return stats
