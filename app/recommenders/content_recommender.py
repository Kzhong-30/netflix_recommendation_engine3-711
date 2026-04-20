import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
import pickle
import os


class ContentRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.feature_names = None
        self.is_trained = False
        self._load_or_train_data()

    def _load_or_train_data(self):
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'movies.csv')
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tfidf_model.pkl')
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        if os.path.exists(model_path) and os.path.exists(data_path):
            self._load_model(model_path, data_path)
        else:
            self._create_sample_data()
            self._train_tfidf()
            self._save_model(model_path, data_path)

    def _create_sample_data(self):
        movies_data = [
            {"title": "The Dark Knight", "genres": ["Action", "Crime", "Drama"], "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.", "vote_average": 8.5, "popularity": 120.5},
            {"title": "Inception", "genres": ["Action", "Sci-Fi", "Thriller"], "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.", "vote_average": 8.3, "popularity": 100.2},
            {"title": "Interstellar", "genres": ["Adventure", "Drama", "Sci-Fi"], "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "vote_average": 8.4, "popularity": 95.8},
            {"title": "The Matrix", "genres": ["Action", "Sci-Fi"], "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.", "vote_average": 8.2, "popularity": 85.3},
            {"title": "Pulp Fiction", "genres": ["Crime", "Drama"], "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.", "vote_average": 8.3, "popularity": 78.9},
            {"title": "Forrest Gump", "genres": ["Drama", "Romance"], "overview": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75.", "vote_average": 8.4, "popularity": 75.6},
            {"title": "The Shawshank Redemption", "genres": ["Drama", "Crime"], "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.", "vote_average": 8.7, "popularity": 80.1},
            {"title": "Fight Club", "genres": ["Drama", "Thriller"], "overview": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.", "vote_average": 8.1, "popularity": 72.4},
            {"title": "The Godfather", "genres": ["Crime", "Drama"], "overview": "The aging patriarch of an organized crime dynasty in postwar New York City transfers control of his clandestine empire to his reluctant youngest son.", "vote_average": 8.6, "popularity": 70.8},
            {"title": "The Lord of the Rings: The Return of the King", "genres": ["Adventure", "Fantasy"], "overview": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring.", "vote_average": 8.5, "popularity": 90.2},
            {"title": "Avengers: Endgame", "genres": ["Action", "Adventure", "Sci-Fi"], "overview": "After the devastating events of Avengers: Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more.", "vote_average": 8.0, "popularity": 130.4},
            {"title": "Spider-Man: Into the Spider-Verse", "genres": ["Animation", "Action", "Adventure"], "overview": "Teen Miles Morales becomes the Spider-Man of his universe, and must join with five spider-powered individuals from other dimensions.", "vote_average": 8.0, "popularity": 85.7},
            {"title": "Parasite", "genres": ["Comedy", "Drama", "Thriller"], "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.", "vote_average": 8.2, "popularity": 65.3},
            {"title": "The Lion King", "genres": ["Animation", "Adventure", "Drama"], "overview": "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.", "vote_average": 7.9, "popularity": 80.5},
            {"title": "Gladiator", "genres": ["Action", "Adventure", "Drama"], "overview": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.", "vote_average": 8.2, "popularity": 68.9},
            {"title": "Titanic", "genres": ["Drama", "Romance"], "overview": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.", "vote_average": 7.8, "popularity": 88.4},
            {"title": "Jurassic Park", "genres": ["Action", "Adventure", "Sci-Fi"], "overview": "A pragmatic paleontologist visiting an almost complete theme park is tasked with protecting a couple of kids after a power failure causes the park's cloned dinosaurs to run loose.", "vote_average": 7.9, "popularity": 75.2},
            {"title": "The Silence of the Lambs", "genres": ["Crime", "Drama", "Thriller"], "overview": "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.", "vote_average": 8.3, "popularity": 67.5},
            {"title": "Goodfellas", "genres": ["Crime", "Drama"], "overview": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners.", "vote_average": 8.4, "popularity": 62.3},
            {"title": "Saving Private Ryan", "genres": ["Drama", "War"], "overview": "Following the Normandy Landings, a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed in action.", "vote_average": 8.2, "popularity": 59.8}
        ]
        
        self.movies_df = pd.DataFrame(movies_data)

    def _train_tfidf(self):
        self.movies_df['content'] = self.movies_df.apply(
            lambda x: f"{x['title']} {' '.join(x['genres'])} {x['overview']}",
            axis=1
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['content'])
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.is_trained = True

    def _save_model(self, model_path: str, data_path: str):
        with open(model_path, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'feature_names': self.feature_names
            }, f)
        self.movies_df.to_csv(data_path, index=False)

    def _load_model(self, model_path: str, data_path: str):
        self.movies_df = pd.read_csv(data_path)
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.feature_names = model_data['feature_names']
        self.is_trained = True

    def get_genre_recommendations(self, preferred_genres: List[str], num_recommendations: int = 10) -> List[Dict[str, Any]]:
        genre_scores = []
        for idx, movie in self.movies_df.iterrows():
            movie_genres = movie['genres']
            if isinstance(movie_genres, str):
                movie_genres = eval(movie_genres)
            
            match_count = len(set(preferred_genres) & set(movie_genres))
            genre_overlap = match_count / len(preferred_genres) if preferred_genres else 0
            
            score = 0.6 * genre_overlap + 0.4 * (movie['vote_average'] / 10.0)
            genre_scores.append((idx, score))
        
        genre_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, score in genre_scores[:num_recommendations]:
            movie = self.movies_df.iloc[idx]
            recommendations.append({
                'title': movie['title'],
                'genres': movie['genres'] if isinstance(movie['genres'], list) else eval(movie['genres']),
                'overview': movie['overview'],
                'vote_average': movie['vote_average'],
                'popularity': movie['popularity'],
                'match_score': float(score),
                'recommendation_reason': f"Matches your preferred genres: {', '.join(preferred_genres)}"
            })
        
        return recommendations

    def get_content_based_vector(self, text: str) -> np.ndarray:
        return self.tfidf_vectorizer.transform([text]).toarray()[0]

    def get_item_feature_matrix(self) -> np.ndarray:
        return self.tfidf_matrix.toarray()
