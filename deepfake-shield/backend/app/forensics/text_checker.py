import os
import httpx
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

async def verify_text_claim(query_text: str) -> dict:
    if not FACT_CHECK_API_KEY:
        return {
            "fact_check_available": False, 
            "matches": [], 
            "note": "API key missing in backend/.env"
        }
        
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query_text, 
        "key": FACT_CHECK_API_KEY, 
        "languageCode": "en"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, params=params, timeout=7.0)
            if resp.status_code != 200:
                return {
                    "fact_check_available": False, 
                    "matches": [], 
                    "error": f"Google API error HTTP {resp.status_code}"
                }
                
            data = resp.json()
            claims = data.get("claims", [])
            
            matches = []
            for c in claims[:3]:
                review = c.get("claimReview", [{}])[0]
                matches.append({
                    "text": c.get("text"),
                    "claimant": c.get("claimant", "Unknown"),
                    "rating": review.get("textualRating", "Unrated"),
                    "publisher": review.get("publisher", {}).get("name", "Unknown Source"),
                    "url": review.get("url", "#")
                })
            return {"fact_check_available": True, "matches": matches}
        except Exception as e:
            return {"fact_check_available": False, "matches": [], "error": str(e)}