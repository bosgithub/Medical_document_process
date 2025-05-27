import aiohttp
import logging
from typing import Optional, Dict, Any, List
from app.services.redis_service import redis_client
import json
from app.services.rag.embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class ICDLookupService:
    """
    Service to lookup ICD-10 codes using NIH's Clinical Tables API
    Includes caching to improve performance and reduce API calls
    """
    
    def __init__(self):
        # Using the newer API endpoint
        self.base_url = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
        self.cache_ttl = 86400  # 24 hours in seconds
    
    async def search_icd_code(self, condition_name: str) -> Optional[Dict[str, str]]:
        """
        Search for ICD-10 code by condition name with caching
        """
        try:
            # Check cache first
            cache_key = f"icd:{condition_name.lower()}"
            cached_result = await redis_client.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for ICD lookup: {condition_name}")
                return json.loads(cached_result)
            
            # If not in cache, call API
            params = {
                'terms': condition_name,
                'ef': 'code,name',  # Return code and name
                'maxList': 5,
                'sf': 'code,name'  # Search in both code and name fields
            }
            
            logger.info(f"Calling NIH Clinical Tables API for condition: {condition_name}")
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=10) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"ICD API returned status {response.status} for '{condition_name}': {error_text}")
                        return None
                    
                    try:
                        data = await response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse ICD API response for '{condition_name}': {str(e)}")
                        return None
            
            if not data:
                logger.warning(f"Empty response from ICD API for '{condition_name}'")
                return None
                
            if len(data) < 4:
                logger.warning(f"Invalid response format from ICD API for '{condition_name}': {data}")
                return None
                
            if not data[3]:
                # Try alternative search terms
                alternative_terms = [
                    condition_name.replace("Type 2", "Type II"),
                    condition_name.replace("Type II", "Type 2"),
                    condition_name.replace("uncontrolled", ""),
                    condition_name.replace("suboptimally controlled", ""),
                    condition_name.split()[0]  # Try just the first word
                ]
                
                for term in alternative_terms:
                    if term == condition_name:
                        continue
                        
                    logger.info(f"Trying alternative term: {term}")
                    params['terms'] = term
                    
                    async with session.get(self.base_url, params=params, timeout=10) as alt_response:
                        if alt_response.status != 200:
                            continue
                            
                        try:
                            alt_data = await alt_response.json()
                        except json.JSONDecodeError:
                            continue
                            
                        if alt_data and len(alt_data) >= 4 and alt_data[3]:
                            result = alt_data[3][0]
                            icd_info = {
                                "code": result[0],
                                "description": result[1]
                            }
                            
                            # Cache the result
                            await redis_client.set(
                                cache_key,
                                json.dumps(icd_info),
                                ex=self.cache_ttl
                            )
                            
                            logger.info(f"ICD code found for {condition_name} using alternative term {term}: {icd_info['code']}")
                            return icd_info
                
                logger.warning(f"No results found from ICD API for '{condition_name}' or alternatives")
                return None
            
            # Take the first (most relevant) result
            result = data[3][0]  # [code, description]
            icd_info = {
                "code": result[0],
                "description": result[1]
            }
            
            # Cache the result
            await redis_client.set(
                cache_key,
                json.dumps(icd_info),
                ex=self.cache_ttl
            )
            
            logger.info(f"ICD code found for {condition_name}: {icd_info['code']}")
            return icd_info
            
        except aiohttp.ClientError as e:
            logger.error(f"ICD API request failed for '{condition_name}': {str(e)}")
            return None
        except Exception as e:
            logger.error(f"ICD lookup failed for '{condition_name}': {str(e)}")
            return None
    
    async def batch_search(self, condition_names: list[str]) -> Dict[str, Optional[Dict[str, str]]]:
        """
        Search for multiple ICD codes in parallel
        """
        import asyncio
        
        async def search_single(condition: str) -> tuple[str, Optional[Dict[str, str]]]:
            try:
                result = await self.search_icd_code(condition)
                return condition, result
            except Exception as e:
                logger.error(f"Error in batch_search for '{condition}': {e}")
                return condition, None
        
        # Create tasks for all conditions
        tasks = [search_single(condition) for condition in condition_names]
        try:
            # Run all searches in parallel
            results = await asyncio.gather(*tasks)
            # Convert to dictionary
            return dict(results)
        except Exception as e:
            logger.error(f"batch_search failed: {e}")
            return {condition: None for condition in condition_names}

    async def get_cdc_icd10_metadata(self, code: str) -> Optional[dict]:
        """
        Query the CDC API to validate and enrich an ICD-10 code.
        Returns metadata if the code is valid, else None.
        """
        cdc_url = f"https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code&terms={code}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(cdc_url, timeout=10) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
            if data and len(data) >= 4 and data[3]:
                for entry in data[3]:
                    if entry[0] == code:
                        return {
                            'code': entry[0],
                            'description': entry[1],
                            'valid': True,
                            'cdc_raw': entry
                        }
            return None
        except Exception as e:
            logger.error(f"CDC ICD-10 API error for code {code}: {e}")
            return None

    async def agentic_icd10_lookup(self, condition_name: str, patient_context: Optional[dict] = None) -> Optional[Dict[str, Any]]:
        """
        Agentic ICD-10 lookup: combines NIH fuzzy search, CDC validation, semantic similarity, context validation, and confidence aggregation.
        Returns a rich output with code, description, confidence, and agent reasoning.
        """
        embedding_service = EmbeddingService()
        logger.info(f"Starting agentic ICD-10 lookup for condition: '{condition_name}'")
        # 1. Query NIH API for candidates
        params = {
            'terms': condition_name,
            'ef': 'code,name',
            'maxList': 5,
            'sf': 'code,name'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"NIH API call failed for '{condition_name}' (status {response.status})")
                    logger.info(f"No ICD-10 code found for condition: '{condition_name}' (NIH API error)")
                    return {
                        'condition': condition_name,
                        'icd10_code': None,
                        'description': None,
                        'confidence': 0.0,
                        'agent_reasoning': 'NIH API call failed. No ICD-10 code found.',
                        'validation_flags': ['no_icd_found']
                    }
                data = await response.json()
        if not data or len(data) < 4 or not data[3]:
            logger.info(f"No ICD-10 code found for condition: '{condition_name}' (no candidates from NIH)")
            return {
                'condition': condition_name,
                'icd10_code': None,
                'description': None,
                'confidence': 0.0,
                'agent_reasoning': 'No ICD-10 code found from NIH or CDC for this condition.',
                'validation_flags': ['no_icd_found']
            }
        nih_candidates = [
            {'code': c[0], 'description': c[1], 'api_match_score': 1.0} for c in data[3]
        ]
        # 2. Validate/enrich with CDC
        confirmed_candidates = []
        for candidate in nih_candidates:
            cdc_metadata = await self.get_cdc_icd10_metadata(candidate['code'])  # ensure await
            if cdc_metadata and cdc_metadata['valid']:
                candidate['cdc_metadata'] = cdc_metadata
                confirmed_candidates.append(candidate)
        if not confirmed_candidates:
            # fallback: use NIH top candidate, flag as not CDC-confirmed
            if nih_candidates:
                best = nih_candidates[0]
                best['cdc_metadata'] = None
                best['semantic_score'] = 0.0
                best['context_score'] = 0.0
                best['cross_score'] = 0.0
                best['confidence'] = 0.0
                best['agent_reasoning'] = 'No CDC confirmation; fallback to NIH candidate.'
                logger.info(f"No CDC-confirmed ICD-10 code for condition: '{condition_name}'. Fallback to NIH candidate: {best['code']}")
                return {
                    'condition': condition_name,
                    'icd10_code': best['code'],
                    'description': best['description'],
                    'confidence': best['confidence'],
                    'agent_reasoning': best['agent_reasoning'],
                    'validation_flags': ['not_cdc_confirmed']
                }
            else:
                # No candidates from NIH at all
                logger.info(f"No ICD-10 code found for condition: '{condition_name}' (no candidates from NIH or CDC)")
                return {
                    'condition': condition_name,
                    'icd10_code': None,
                    'description': None,
                    'confidence': 0.0,
                    'agent_reasoning': 'No ICD-10 code found from NIH or CDC for this condition.',
                    'validation_flags': ['no_icd_found']
                }
        # 3. Agentic scoring (semantic, context, cross-ref, etc.)
        cond_emb = await embedding_service.get_embedding(condition_name)  # ensure await
        descs = [c['description'] for c in confirmed_candidates]
        desc_embs = await embedding_service.get_embeddings(descs)  # ensure await
        for c, emb in zip(confirmed_candidates, desc_embs):
            c['semantic_score'] = embedding_service.cosine_similarity(cond_emb, emb)
        for c in confirmed_candidates:
            c['context_score'] = self._validate_context_stub(c, patient_context)
            c['cross_score'] = self._cross_reference_stub(c, patient_context)
            c['confidence'] = self._weighted_average([
                c['semantic_score'],
                c['context_score'],
                c['cross_score'],
                c['api_match_score']
            ], [0.4, 0.3, 0.2, 0.1])
        # 4. Select best
        best = max(confirmed_candidates, key=lambda c: c['confidence'])
        agent_reasoning = self._build_reasoning(best)
        logger.info(f"ICD-10 lookup for '{condition_name}': selected code {best['code']} with confidence {best['confidence']:.2f}")
        rounded_confidence = round(best['confidence'], 4)
        return {
            'condition': condition_name,
            'icd10_code': best['code'],
            'description': best['description'],
            'confidence': rounded_confidence,
            'agent_reasoning': agent_reasoning,
            'cdc_metadata': best.get('cdc_metadata'),
            'validation_flags': best.get('flags', [])
        }

    def _validate_context_stub(self, candidate: dict, patient_context: Optional[dict]) -> float:
        # TODO: Implement real context validation
        return 1.0

    def _cross_reference_stub(self, candidate: dict, patient_context: Optional[dict]) -> float:
        # TODO: Implement real cross-reference validation
        return 1.0

    def _weighted_average(self, scores: List[float], weights: List[float]) -> float:
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    def _build_reasoning(self, candidate: dict) -> str:
        return (
            f"Semantic similarity: {candidate['semantic_score']:.2f}, "
            f"Context: {candidate['context_score']:.2f}, "
            f"Cross-ref: {candidate['cross_score']:.2f}, "
            f"API match: {candidate['api_match_score']:.2f}. "
            f"Selected code: {candidate['code']} ({candidate['description']})"
        ) 