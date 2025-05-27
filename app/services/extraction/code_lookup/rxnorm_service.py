import aiohttp
import logging
from typing import Optional, Dict, Any, List
from app.services.redis_service import redis_client
import json

logger = logging.getLogger(__name__)

class RxNormLookupService:
    """
    Service to lookup RxNorm codes using NIH's RxNav API
    Includes caching to improve performance and reduce API calls
    """
    def __init__(self):
        self.base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
        self.cache_ttl = 86400  # 24 hours in seconds

    async def lookup_rxnorm_code(self, medication_name: str) -> Optional[Dict[str, Any]]:
        """
        Lookup RxNorm code for a medication name with caching and confidence scoring.
        Returns a dict with rxnorm_code, description, and confidence.
        """
        try:
            cache_key = f"rxnorm:{medication_name.lower()}"
            cached_result = await redis_client.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for RxNorm lookup: {medication_name}")
                return json.loads(cached_result)

            params = {"name": medication_name}
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=10) as response:
                    if response.status != 200:
                        logger.error(f"RxNorm API returned status {response.status} for '{medication_name}'")
                        return None
                    data = await response.json()

            # Parse RxNorm rxcui
            rxcui = None
            if data and "idGroup" in data and "rxnormId" in data["idGroup"]:
                ids = data["idGroup"]["rxnormId"]
                if ids:
                    rxcui = ids[0]
            if not rxcui:
                logger.warning(f"No RxNorm code found for '{medication_name}'")
                result = {
                    "name": medication_name,
                    "rxnorm_code": None,
                    "rxnorm_description": None,
                    "confidence": 0.0
                }
                await redis_client.set(cache_key, json.dumps(result), ex=self.cache_ttl)
                return result

            # Optionally, get the canonical name/description
            desc_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
            rxnorm_description = None
            async with aiohttp.ClientSession() as session:
                async with session.get(desc_url, timeout=10) as desc_response:
                    if desc_response.status == 200:
                        desc_data = await desc_response.json()
                        if desc_data and "properties" in desc_data and "name" in desc_data["properties"]:
                            rxnorm_description = desc_data["properties"]["name"]

            result = {
                "name": medication_name,
                "rxnorm_code": rxcui,
                "rxnorm_description": rxnorm_description or medication_name,
                "confidence": 1.0 if rxnorm_description else 0.8
            }
            await redis_client.set(cache_key, json.dumps(result), ex=self.cache_ttl)
            logger.info(f"RxNorm code found for {medication_name}: {rxcui}")
            return result
        except Exception as e:
            logger.error(f"RxNorm lookup failed for '{medication_name}': {str(e)}")
            return None

    async def batch_lookup(self, medication_names: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Lookup RxNorm codes for multiple medications in parallel.
        Returns a dict mapping medication name to lookup result.
        """
        import asyncio
        async def lookup_single(med: str) -> tuple[str, Optional[Dict[str, Any]]]:
            try:
                result = await self.lookup_rxnorm_code(med)
                return med, result
            except Exception as e:
                logger.error(f"Error in batch_lookup for '{med}': {e}")
                return med, None
        tasks = [lookup_single(med) for med in medication_names]
        try:
            results = await asyncio.gather(*tasks)
            return dict(results)
        except Exception as e:
            logger.error(f"batch_lookup failed: {e}")
            return {med: None for med in medication_names} 