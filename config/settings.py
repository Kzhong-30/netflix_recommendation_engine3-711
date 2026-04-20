import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """configuration settings for the recommendation engine"""

    # API keys
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./recommendations.db")

    # API SETTINGS
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    # TMDB API settings
    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

    # recommendation settings
    DEFAULT_RECOMMENDATION_COUNT = 10
    MIN_RATINGS_FOR_COLLAB_FILTER = 5

    # Data processing
    BATCH_SIZE = 100
    MAX_RETRIES = 3

# Create global settings instance
settings = Settings()

# Validate required settings
def validate_config():
    """Check if all required configuration is present"""
    if not settings.TMDB_API_KEY:
        raise ValueError("TMDB_API_KEY environment variable is required")
    print("✅ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    