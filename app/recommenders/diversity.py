import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def inject_diversity(
    candidates: List[Dict[str, Any]],
    tfidf_matrix: np.ndarray,
    movies_df: pd.DataFrame,
    num_recommendations: int = 10,
    lambda_param: float = 0.7,
    max_genre_repetition: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Inject diversity into recommendations using Maximal Marginal Relevance (MMR).
    
    MMR balances relevance and diversity by selecting items that are both relevant
    to the query and different from already selected items.
    
    Args:
        candidates: List of candidate movies with scores
        tfidf_matrix: TF-IDF feature matrix for similarity computation
        movies_df: DataFrame with movie information
        num_recommendations: Number of recommendations to return
        lambda_param: Trade-off between relevance and diversity (0-1)
                     Higher values prioritize relevance, lower values prioritize diversity
        max_genre_repetition: Maximum allowed genre repetition ratio (0-1)
    
    Returns:
        List of diverse recommendations
    
    Algorithm:
        MMR = lambda * Sim(item, query) - (1-lambda) * max(Sim(item, selected))
    """
    if len(candidates) == 0:
        return []
    
    if len(candidates) <= num_recommendations:
        return candidates
    
    selected = []
    selected_indices = []
    selected_genres = []
    
    candidates_sorted = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
    
    first_item = candidates_sorted[0]
    selected.append(first_item)
    selected_indices.append(first_item.get('index', 0))
    selected_genres.extend(first_item.get('genres', []))
    
    remaining = candidates_sorted[1:]
    
    while len(selected) < num_recommendations and remaining:
        best_score = -np.inf
        best_item = None
        best_idx = -1
        
        for idx, item in enumerate(remaining):
            relevance_score = item.get('score', 0)
            
            item_index = item.get('index', -1)
            if item_index == -1 or item_index >= tfidf_matrix.shape[0]:
                diversity_penalty = 0
            else:
                item_vector = tfidf_matrix[item_index:item_index+1]
                selected_vectors = tfidf_matrix[selected_indices]
                
                similarities = cosine_similarity(item_vector, selected_vectors)[0]
                diversity_penalty = np.max(similarities)
            
            mmr_score = lambda_param * relevance_score - (1 - lambda_param) * diversity_penalty
            
            item_genres = item.get('genres', [])
            if selected_genres:
                genre_counts = {}
                for g in selected_genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                
                max_genre_count = max(genre_counts.values()) if genre_counts else 0
                current_genre_ratio = max_genre_count / len(selected) if selected else 0
                
                new_genre_overlap = sum(1 for g in item_genres if g in selected_genres)
                if current_genre_ratio >= max_genre_repetition and new_genre_overlap > 0:
                    mmr_score *= 0.5
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
                best_idx = idx
        
        if best_item:
            selected.append(best_item)
            selected_indices.append(best_item.get('index', 0))
            selected_genres.extend(best_item.get('genres', []))
            remaining.pop(best_idx)
    
    for item in selected:
        item['mmr_applied'] = True
        item['diversity_score'] = item.get('score', 0)
    
    return selected


def calculate_diversity_metrics(recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate diversity metrics for a list of recommendations.
    
    Args:
        recommendations: List of recommended items with genres
    
    Returns:
        Dictionary with diversity metrics
    """
    if not recommendations:
        return {'genre_diversity': 0, 'unique_genres': 0, 'genre_repetition': 0}
    
    all_genres = []
    for item in recommendations:
        genres = item.get('genres', [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.replace('[', '').replace(']', '').replace("'", '').split(',')]
        all_genres.extend(genres)
    
    if not all_genres:
        return {'genre_diversity': 0, 'unique_genres': 0, 'genre_repetition': 0}
    
    unique_genres = set(all_genres)
    genre_counts = {}
    for g in all_genres:
        genre_counts[g] = genre_counts.get(g, 0) + 1
    
    max_count = max(genre_counts.values())
    total_count = len(all_genres)
    
    genre_diversity = len(unique_genres) / len(recommendations) if recommendations else 0
    genre_repetition = max_count / total_count if total_count > 0 else 0
    
    return {
        'genre_diversity': round(genre_diversity, 3),
        'unique_genres': len(unique_genres),
        'genre_repetition': round(genre_repetition, 3),
        'total_genres': total_count,
        'genre_distribution': genre_counts
    }


def validate_diversity_constraint(
    recommendations: List[Dict[str, Any]],
    max_genre_repetition: float = 0.3
) -> bool:
    """
    Validate that recommendations meet the diversity constraint.
    
    Args:
        recommendations: List of recommended items
        max_genre_repetition: Maximum allowed genre repetition ratio
    
    Returns:
        True if constraint is satisfied, False otherwise
    """
    metrics = calculate_diversity_metrics(recommendations)
    return metrics['genre_repetition'] <= max_genre_repetition
