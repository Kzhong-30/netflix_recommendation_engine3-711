import requests
import time
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_genres():
    """Test genres endpoint"""
    print("\n=== Testing Genres Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/genres")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total genres: {data['total_count']}")
    print(f"Sample genres: {json.dumps(data['genres'][:5], indent=2)}")
    return response.status_code == 200


def test_cold_start_recommendation():
    """Test cold-start recommendation endpoint"""
    print("\n=== Testing Cold-Start Recommendation ===")
    
    test_cases = [
        {
            "name": "Action/Adventure User",
            "genres": ["Action", "Adventure"],
        },
        {
            "name": "Comedy/Drama User",
            "genres": ["Comedy", "Drama", "Romance"],
        },
        {
            "name": "Sci-Fi/Horror User",
            "genres": ["Science Fiction", "Horror", "Thriller"],
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        payload = {
            "preferred_genres": test_case["genres"],
            "num_recommendations": 10,
            "min_rating": 6.0,
            "diversity_lambda": 0.7
        }
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/recommend/cold-start",
            json=payload
        )
        inference_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"Strategy: {data['strategy']}")
            print(f"Response time: {data['metadata']['inference_time_ms']}ms (total: {inference_time:.2f}ms)")
            print(f"Candidate pool: {data['metadata']['candidate_pool_size']}")
            
            print(f"\nDiversity Metrics:")
            div_metrics = data['diversity_metrics']
            print(f"  - Genre diversity: {div_metrics['genre_diversity']}")
            print(f"  - Unique genres: {div_metrics['unique_genres']}")
            print(f"  - Genre repetition: {div_metrics['genre_repetition']} ({'PASS' if div_metrics['genre_repetition'] <= 0.3 else 'FAIL'})")
            
            print(f"\nTop 5 Recommendations:")
            for i, rec in enumerate(data['recommendations'][:5], 1):
                print(f"  {i}. {rec['title']} (Rating: {rec['vote_average']}, Genres: {rec['genres']})")
            
            results.append({
                "name": test_case["name"],
                "success": True,
                "response_time_ms": inference_time,
                "genre_repetition": div_metrics['genre_repetition'],
                "diversity_pass": div_metrics['genre_repetition'] <= 0.3
            })
        else:
            print(f"Error: {response.status_code} - {response.text}")
            results.append({
                "name": test_case["name"],
                "success": False
            })
    
    return results


def test_performance():
    """Test response time performance"""
    print("\n=== Testing Performance ===")
    
    payload = {
        "preferred_genres": ["Action", "Adventure", "Sci-Fi"],
        "num_recommendations": 10
    }
    
    times = []
    for i in range(5):
        start = time.time()
        response = requests.post(f"{BASE_URL}/api/recommend/cold-start", json=payload)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"Request {i+1}: {elapsed:.2f}ms")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage response time: {avg_time:.2f}ms")
    print(f"Performance check: {'PASS' if avg_time < 500 else 'FAIL'} (target: <500ms)")
    
    return avg_time < 500


def main():
    print("=" * 60)
    print("Cold-Start Recommendation API Test Suite")
    print("=" * 60)
    
    try:
        health_ok = test_health()
        if not health_ok:
            print("\n❌ Health check failed. Is the API running?")
            return
        
        test_genres()
        results = test_cold_start_recommendation()
        performance_ok = test_performance()
        
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        successful = sum(1 for r in results if r.get('success', False))
        diversity_pass = sum(1 for r in results if r.get('diversity_pass', False))
        
        print(f"Recommendation tests: {successful}/{len(results)} passed")
        print(f"Diversity constraint: {diversity_pass}/{len(results)} passed (<30% repetition)")
        print(f"Performance check: {'PASS' if performance_ok else 'FAIL'} (<500ms)")
        
        all_passed = successful == len(results) and diversity_pass == len(results) and performance_ok
        print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Please start the server first:")
        print("   uvicorn api.cold_start_api:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
