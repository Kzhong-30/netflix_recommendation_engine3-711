import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cold_start import ColdStartRecommender
from app.recommenders.diversity import inject_diversity, calculate_diversity_metrics, validate_diversity_constraint


class TestColdStartRecommender(unittest.TestCase):
    """Test cases for ColdStartRecommender"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        np.random.seed(42)
        
        genres_list = [
            "Action", "Adventure", "Animation", "Comedy", "Crime",
            "Drama", "Fantasy", "Horror", "Romance", "Science Fiction",
            "Thriller", "War", "Western"
        ]
        
        movie_titles = [
            "The Shawshank Redemption", "The Godfather", "The Dark Knight",
            "Pulp Fiction", "Forrest Gump", "Inception", "The Matrix",
            "Goodfellas", "Fight Club", "The Lord of the Rings",
            "Star Wars", "Interstellar", "The Prestige", "Gladiator",
            "The Departed", "Saving Private Ryan", "The Green Mile",
            "Parasite", "Joker", "Avengers: Endgame", "Titanic",
            "Avatar", "The Lion King", "Jurassic Park", "The Avengers",
            "Spider-Man: No Way Home", "Top Gun: Maverick", "Oppenheimer",
            "Dune", "The Batman", "Black Panther", "Wonder Woman"
        ]
        
        cls.movies_df = pd.DataFrame({
            'id': range(1, len(movie_titles) + 1),
            'title': movie_titles,
            'genres': [list(np.random.choice(genres_list, size=np.random.randint(1, 4), replace=False)) for _ in movie_titles],
            'overview': ["An epic tale of adventure and excitement."] * len(movie_titles),
            'vote_average': np.random.uniform(6.0, 9.0, len(movie_titles)),
            'vote_count': np.random.randint(100, 10000, len(movie_titles)),
            'release_date': ['2020-01-01'] * len(movie_titles)
        })
        
        cls.recommender = ColdStartRecommender()
        cls.recommender.fit(cls.movies_df)
    
    def test_recommender_is_fitted(self):
        """Test that recommender is properly fitted"""
        self.assertTrue(self.recommender.is_fitted)
        self.assertIsNotNone(self.recommender.tfidf_matrix)
        self.assertIsNotNone(self.recommender.tfidf_vectorizer)
    
    def test_recommend_returns_correct_count(self):
        """Test that recommend returns correct number of recommendations"""
        recommendations, metadata = self.recommender.recommend(
            preferred_genres=["Action", "Adventure"],
            num_recommendations=10
        )
        
        self.assertEqual(len(recommendations), 10)
        self.assertIn('strategy', metadata)
        self.assertIn('inference_time_ms', metadata)
    
    def test_recommend_response_time(self):
        """Test that recommendation response time is under 500ms"""
        import time
        
        start = time.time()
        recommendations, metadata = self.recommender.recommend(
            preferred_genres=["Action", "Drama"],
            num_recommendations=10
        )
        elapsed_ms = (time.time() - start) * 1000
        
        self.assertLess(elapsed_ms, 500, f"Response time {elapsed_ms}ms exceeds 500ms limit")
    
    def test_recommend_with_empty_genres(self):
        """Test recommendation with empty genre list"""
        recommendations, metadata = self.recommender.recommend(
            preferred_genres=[],
            num_recommendations=5
        )
        
        self.assertEqual(len(recommendations), 5)
    
    def test_genre_statistics(self):
        """Test genre statistics retrieval"""
        stats = self.recommender.get_genre_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)


class TestDiversityInjection(unittest.TestCase):
    """Test cases for diversity injection"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        np.random.seed(42)
        
        cls.movies_df = pd.DataFrame({
            'id': range(1, 21),
            'title': [f"Movie {i}" for i in range(1, 21)],
            'genres': [
                ["Action", "Adventure"] if i < 7 else
                ["Comedy", "Romance"] if i < 13 else
                ["Horror", "Thriller"]
                for i in range(1, 21)
            ],
            'vote_average': np.random.uniform(6.0, 9.0, 20)
        })
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        cls.movies_df['content'] = cls.movies_df['genres'].apply(lambda x: ' '.join(x))
        cls.vectorizer = TfidfVectorizer()
        cls.tfidf_matrix = cls.vectorizer.fit_transform(cls.movies_df['content'])
        
        cls.candidates = [
            {
                'index': i,
                'title': row['title'],
                'genres': row['genres'],
                'vote_average': row['vote_average'],
                'score': np.random.uniform(0.5, 1.0)
            }
            for i, row in cls.movies_df.iterrows()
        ]
    
    def test_inject_diversity_returns_correct_count(self):
        """Test that diversity injection returns correct number of items"""
        result = inject_diversity(
            candidates=self.candidates,
            tfidf_matrix=self.tfidf_matrix,
            movies_df=self.movies_df,
            num_recommendations=10
        )
        
        self.assertEqual(len(result), 10)
    
    def test_diversity_constraint_satisfied(self):
        """Test that genre repetition is below 30%"""
        result = inject_diversity(
            candidates=self.candidates,
            tfidf_matrix=self.tfidf_matrix,
            movies_df=self.movies_df,
            num_recommendations=10,
            max_genre_repetition=0.3
        )
        
        metrics = calculate_diversity_metrics(result)
        
        self.assertLessEqual(
            metrics['genre_repetition'], 
            0.3,
            f"Genre repetition {metrics['genre_repetition']} exceeds 30% limit"
        )
    
    def test_calculate_diversity_metrics(self):
        """Test diversity metrics calculation"""
        test_recommendations = [
            {'genres': ['Action', 'Adventure']},
            {'genres': ['Action', 'Comedy']},
            {'genres': ['Drama', 'Romance']},
            {'genres': ['Horror', 'Thriller']},
            {'genres': ['Action', 'Thriller']}
        ]
        
        metrics = calculate_diversity_metrics(test_recommendations)
        
        self.assertIn('genre_diversity', metrics)
        self.assertIn('unique_genres', metrics)
        self.assertIn('genre_repetition', metrics)
        self.assertGreater(metrics['unique_genres'], 0)
    
    def test_validate_diversity_constraint(self):
        """Test diversity constraint validation"""
        good_recommendations = [
            {'genres': ['Action']},
            {'genres': ['Comedy']},
            {'genres': ['Drama']},
            {'genres': ['Horror']},
            {'genres': ['Romance']}
        ]
        
        bad_recommendations = [
            {'genres': ['Action']},
            {'genres': ['Action']},
            {'genres': ['Action']},
            {'genres': ['Action']},
            {'genres': ['Action']}
        ]
        
        self.assertTrue(validate_diversity_constraint(good_recommendations, 0.3))
        self.assertFalse(validate_diversity_constraint(bad_recommendations, 0.3))


class TestAccuracy(unittest.TestCase):
    """Test recommendation accuracy for cold-start users"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with known genre mappings"""
        np.random.seed(42)
        
        cls.movies_df = pd.DataFrame({
            'id': range(1, 51),
            'title': [f"Movie {i}" for i in range(1, 51)],
            'genres': [
                ["Action", "Adventure"] if i <= 15 else
                ["Comedy", "Romance"] if i <= 30 else
                ["Horror", "Thriller"] if i <= 40 else
                ["Drama"]
                for i in range(1, 51)
            ],
            'overview': ["A great movie with exciting plot."] * 50,
            'vote_average': [7.0 + (i % 30) * 0.05 for i in range(50)],
            'vote_count': [1000 + i * 100 for i in range(50)],
            'release_date': ['2020-01-01'] * 50
        })
        
        cls.recommender = ColdStartRecommender()
        cls.recommender.fit(cls.movies_df)
    
    def test_accuracy_above_40_percent(self):
        """Test that accuracy for first 10 recommendations is above 40%"""
        preferred_genres = ["Action", "Adventure"]
        
        recommendations, _ = self.recommender.recommend(
            preferred_genres=preferred_genres,
            num_recommendations=10
        )
        
        relevant_count = 0
        for rec in recommendations:
            rec_genres = rec.get('genres', [])
            if any(g in preferred_genres for g in rec_genres):
                relevant_count += 1
        
        accuracy = relevant_count / len(recommendations)
        
        self.assertGreater(
            accuracy, 
            0.4,
            f"Accuracy {accuracy*100:.1f}% is below 40% threshold"
        )
    
    def test_multiple_genre_preferences(self):
        """Test recommendations for multiple genre preferences"""
        test_cases = [
            (["Action", "Adventure"], "Action/Adventure"),
            (["Comedy", "Romance"], "Comedy/Romance"),
            (["Horror", "Thriller"], "Horror/Thriller"),
        ]
        
        for genres, name in test_cases:
            with self.subTest(genres=name):
                recommendations, _ = self.recommender.recommend(
                    preferred_genres=genres,
                    num_recommendations=10
                )
                
                self.assertEqual(len(recommendations), 10)
                
                relevant_count = sum(
                    1 for rec in recommendations
                    if any(g in genres for g in rec.get('genres', []))
                )
                
                accuracy = relevant_count / len(recommendations)
                self.assertGreater(accuracy, 0.4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
