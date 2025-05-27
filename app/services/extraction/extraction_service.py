import logging
import re
import aiohttp
from typing import Dict, List, Optional
from app.services.llm.llm_service import LLMService
from app.services.extraction.code_lookup import ICDLookupService
from app.services.extraction.extraction_schemas import (
    StructuredMedicalData,
    ExtractedPatientInfo,
    ExtractedCondition,
    ExtractedMedication,
    ExtractedVitalSign,
    ExtractedLabResult,
    ExtractedPlan
)
from app.services.extraction.code_lookup.rxnorm_service import RxNormLookupService

logger = logging.getLogger(__name__)

class ExtractionService:
    """
    Main service for extracting structured medical data from notes.
    Coordinates between LLM extraction and medical code lookups.
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.icd_service = ICDLookupService()
        self.rxnorm_service = RxNormLookupService()
        
        # Common condition patterns
        self.condition_patterns = [
            r'(?:diagnosed with|history of|hx of|suffering from)\s+([^,.]+)',
            r'(?:possible|probable|suspected)\s+([^,.]+)',
            r'(?:BMI\s+\d+\.\d+\s*\(([^)]+)\))',
            r'(?:family hx of|family history of)\s+([^,.]+)',
            r'(?:screening for|screening initiated for)\s+([^,.]+)'
        ]
        
        # Common medication patterns
        self.medication_patterns = [
            r'(?:prescribed|taking|on)\s+([^,.]+)',
            r'(?:Rx|prescription):\s+([^,.]+)',
            r'(?:medication|meds):\s+([^,.]+)'
        ]
        
        # Vital sign patterns
        self.vital_patterns = {
            'Blood Pressure': r'BP:\s*(\d+/\d+\s*mmHg)',
            'Heart Rate': r'HR:\s*(\d+\s*bpm)',
            'Respiratory Rate': r'RR:\s*(\d+\s*breaths/min)',
            'Temperature': r'Temp:\s*(\d+\.\d+°F)',
            'Height': r'Ht:\s*([\d\'\"]+)',
            'Weight': r'Wt:\s*(\d+\s*lbs)'
        }
        
        # Lab test patterns
        self.lab_patterns = [
            r'Labs ordered:\s*([^,.]+)',
            r'Lab results:\s*([^,.]+)',
            r'Test results:\s*([^,.]+)'
        ]
        
        # Procedure patterns
        self.procedure_patterns = [
            r'(?:administered|performed|completed)\s+([^,.]+)',
            r'(?:procedure|treatment):\s+([^,.]+)'
        ]
        
        logger.info("ExtractionService initialized")
    
    def _extract_conditions(self, note: str) -> List[str]:
        """Extract medical conditions from the note"""
        conditions = set()
        
        # Look for conditions in Assessment section
        assessment_match = re.search(r'A:\s*(.*?)(?=\n\nP:|$)', note, re.DOTALL | re.IGNORECASE)
        if assessment_match:
            assessment_text = assessment_match.group(1)
            
            # Apply each pattern
            for pattern in self.condition_patterns:
                matches = re.finditer(pattern, assessment_text, re.IGNORECASE)
                for match in matches:
                    condition = match.group(1).strip()
                    # Clean up the condition name
                    condition = re.sub(r'\s+', ' ', condition)
                    condition = condition.strip('()[]{}')
                    if condition and len(condition) > 2:  # Avoid single words or very short phrases
                        conditions.add(condition)
        
        return list(conditions)
    
    def _extract_medications(self, note: str) -> List[str]:
        """Extract medications from the note"""
        medications = set()
        
        # Look for medications in Plan section
        plan_match = re.search(r'P:\s*(.*?)(?=\n\nSigned:|$)', note, re.DOTALL | re.IGNORECASE)
        if plan_match:
            plan_text = plan_match.group(1)
            
            # Apply each pattern
            for pattern in self.medication_patterns:
                matches = re.finditer(pattern, plan_text, re.IGNORECASE)
                for match in matches:
                    medication = match.group(1).strip()
                    # Clean up the medication name
                    medication = re.sub(r'\s+', ' ', medication)
                    medication = medication.strip('()[]{}')
                    if medication and len(medication) > 2:
                        medications.add(medication)
        
        return list(medications)
    
    def _extract_vital_signs(self, note: str) -> Dict[str, str]:
        """Extract vital signs from the note"""
        vitals = {}
        
        # Look for vitals section
        vitals_match = re.search(r'Vitals:\s*(.*?)(?=\n\n|$)', note, re.DOTALL | re.IGNORECASE)
        if vitals_match:
            vitals_text = vitals_match.group(1)
            
            # Apply each pattern
            for name, pattern in self.vital_patterns.items():
                match = re.search(pattern, vitals_text, re.IGNORECASE)
                if match:
                    vitals[name] = match.group(1).strip()
        
        return vitals
    
    def _extract_lab_tests(self, note: str) -> List[str]:
        """Extract lab tests from the note"""
        lab_tests = set()
        
        # Look for lab tests in Objective section
        objective_match = re.search(r'O:\s*(.*?)(?=\n\nA:|$)', note, re.DOTALL | re.IGNORECASE)
        if objective_match:
            objective_text = objective_match.group(1)
            
            # Apply each pattern
            for pattern in self.lab_patterns:
                matches = re.finditer(pattern, objective_text, re.IGNORECASE)
                for match in matches:
                    tests = match.group(1).strip()
                    # Split on common delimiters
                    for test in re.split(r'[,;]', tests):
                        test = test.strip()
                        if test and len(test) > 2:
                            lab_tests.add(test)
        
        return list(lab_tests)
    
    def _extract_procedures(self, note: str) -> List[str]:
        """Extract procedures from the note"""
        procedures = set()
        
        # Look for procedures in Plan section
        plan_match = re.search(r'P:\s*(.*?)(?=\n\nSigned:|$)', note, re.DOTALL | re.IGNORECASE)
        if plan_match:
            plan_text = plan_match.group(1)
            
            # Apply each pattern
            for pattern in self.procedure_patterns:
                matches = re.finditer(pattern, plan_text, re.IGNORECASE)
                for match in matches:
                    procedure = match.group(1).strip()
                    # Clean up the procedure name
                    procedure = re.sub(r'\s+', ' ', procedure)
                    procedure = procedure.strip('()[]{}')
                    if procedure and len(procedure) > 2:
                        procedures.add(procedure)
        
        return list(procedures)
    
    def postprocess_llm_entities(self, entities):
        """
        Post-process LLM output to reclassify and standardize entities.
        - Separates conditions, symptoms, and procedures
        - Handles 'status-post' and similar phrases
        - Standardizes terms
        """
        conditions, symptoms, procedures = [], [], []
        # Example lists for reclassification
        procedure_keywords = ["repair", "immunization", "arthroscopy", "arthroscopic", "injection", "surgery", "ectomy", "scopy"]
        symptom_list = ["pain", "stiffness", "discomfort", "swelling", "nausea", "headache", "fatigue", "fever", "cough", "dizziness", "itching", "rash"]

        # Reclassify conditions
        for cond in entities.get("conditions", []):
            cond_lower = cond.lower()
            if any(word in cond_lower for word in procedure_keywords):
                procedures.append(cond)
            elif any(symptom in cond_lower for symptom in symptom_list):
                symptoms.append(cond)
            elif cond_lower.startswith("status-post") or cond_lower.startswith("s/p"):
                # Extract procedure and infer condition
                proc = cond_lower.replace("status-post", "").replace("s/p", "").strip()
                procedures.append(proc)
                # Map procedure to likely condition (custom mapping or LLM call)
                if "meniscal repair" in proc:
                    conditions.append("Meniscal tear")
            else:
                conditions.append(cond)

        # Add original symptoms and procedures
        symptoms.extend(entities.get("symptoms", []))
        procedures.extend(entities.get("procedures", []))

        # Standardize and deduplicate
        def standardize_term(term):
            return term.strip().capitalize()
        conditions = list(set([standardize_term(c) for c in conditions if c]))
        symptoms = list(set([standardize_term(s) for s in symptoms if s]))
        procedures = list(set([standardize_term(p) for p in procedures if p]))

        # Return post-processed entities
        return {
            "conditions": conditions,
            "symptoms": symptoms,
            "procedures": procedures,
            # Pass through other fields if present
            **{k: v for k, v in entities.items() if k not in ["conditions", "symptoms", "procedures"]}
        }

    async def suggest_icd10_via_nih(self, condition_name, threshold=0.6, top_n=10):
        embedding_service = self.llm_service.embedding_service
        # Synonym mapping for common terms
        SYNONYM_MAP = {
            "low hdl": "hypoalphalipoproteinemia",
            "low hdl": "low high-density lipoprotein",
            "low hdl": "low high density lipoprotein",
            "high cholesterol": "hypercholesterolemia",
            "high blood pressure": "hypertension",
            "high triglycerides": "hypertriglyceridemia",
            # Add more as needed
        }
        query_term = SYNONYM_MAP.get(condition_name.lower(), condition_name)
        # Query NIH API for top N candidates
        params = {
            'terms': query_term,
            'ef': 'code,name',
            'maxList': 30,  # Get more candidates for similarity search
            'sf': 'code,name'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(self.icd_service.base_url, params=params, timeout=10) as response:
                if response.status != 200:
                    return None
                data = await response.json()
        if not data or len(data) < 4 or not data[3]:
            return None
        candidates = [{"code": c[0], "description": c[1]} for c in data[3]]

        # Compute embeddings
        cond_emb = await embedding_service.get_embedding(condition_name)
        descs = [c["description"] for c in candidates]
        desc_embs = await embedding_service.get_embeddings(descs)

        # Compute similarities
        scored_candidates = []
        for candidate, emb in zip(candidates, desc_embs):
            score = embedding_service.cosine_similarity(cond_emb, emb)
            scored_candidates.append({
                "code": candidate["code"],
                "description": candidate["description"],
                "similarity_score": score
            })
        # Sort by similarity
        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_suggestions = scored_candidates[:top_n]
        # Always return top suggestions, even if below threshold
        return {"top_suggestions": top_suggestions}
        # TODO: Fallback to LLM for mapping if all else fails

    async def llm_icd10_fallback(self, condition_name):
        # Use the LLM to suggest the most likely ICD-10 code and description for the condition
        prompt = f"""
        Given the following medical condition, suggest the most likely ICD-10 code and its description. Return your answer as a JSON object with keys 'code' and 'description'.
        
        Condition: {condition_name}
        """
        try:
            response = await self.llm_service.generate_text(prompt)
            # Try to parse the LLM's response as JSON
            import json
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"LLM fallback ICD-10 mapping failed for '{condition_name}': {str(e)}")
            return None

    async def extract_medical_data(self, note: str, model: Optional[str] = None) -> StructuredMedicalData:
        """Extract all medical data from the note and return as StructuredMedicalData (LLM-only extraction, robust to missing/ambiguous fields)"""
        try:
            # LLM extraction for all entities
            entities = await self.llm_service.extract_entities(note, model=model or "gpt-3.5-turbo-zero-temp")

            # Post-process entities for better classification
            entities = self.postprocess_llm_entities(entities)

            patient_info = entities.get("patient_info")
            chief_complaint = entities.get("chief_complaint")
            conditions = entities.get("conditions", [])
            medications = entities.get("medications", [])
            medications = [
                m if isinstance(m, dict) else {"name": m}
                for m in medications
            ]
            # Lowercase all medication names for consistency
            for m in medications:
                if "name" in m and isinstance(m["name"], str):
                    m["name"] = m["name"].lower()
            vital_signs = entities.get("vital_signs", [])
            lab_results = entities.get("lab_results", [])
            procedures = entities.get("procedures", [])
            plan_items = entities.get("plan_items", [])

            # ICD-10 lookup for conditions (still required)
            extracted_conditions = []
            for condition in conditions:
                icd_result = await self.icd_service.agentic_icd10_lookup(condition, patient_info)
                confidence = icd_result["confidence"] if icd_result and icd_result.get("confidence") is not None else 0.0
                confidence = min(float(confidence), 1.0)
                cond_entry = {
                    "name": condition,
                    "icd_code": icd_result["icd10_code"] if icd_result else None,
                    "icd_description": icd_result["description"] if icd_result else None,
                    "confidence": confidence
                }
                # Feedback loop for unmapped/low-confidence
                if not cond_entry["icd_code"] or confidence < 0.5:
                    suggestion = await self.suggest_icd10_via_nih(condition)
                    if suggestion and suggestion["top_suggestions"]:
                        cond_entry["top_suggestions"] = suggestion["top_suggestions"]
                        # Auto-assign the best suggestion if main code is missing
                        best = suggestion["top_suggestions"][0]
                        cond_entry["icd_code"] = best["code"]
                        cond_entry["icd_description"] = best["description"]
                        cond_entry["similarity_score"] = best["similarity_score"]
                # If still no code, use LLM fallback
                if not cond_entry["icd_code"]:
                    llm_suggestion = await self.llm_icd10_fallback(condition)
                    if llm_suggestion and llm_suggestion.get("code"):
                        cond_entry["icd_code"] = llm_suggestion["code"]
                        cond_entry["icd_description"] = llm_suggestion.get("description")
                        cond_entry["llm_fallback"] = True
                extracted_conditions.append(cond_entry)

            # RxNorm lookup for medications
            extracted_medications = []
            for med in medications:
                med_name = med.get("name")
                rxnorm_result = await self.rxnorm_service.lookup_rxnorm_code(med_name)
                med_entry = med.copy()
                if rxnorm_result:
                    med_entry["rxnorm_code"] = rxnorm_result.get("rxnorm_code")
                    med_entry["rxnorm_description"] = rxnorm_result.get("rxnorm_description")
                    med_entry["confidence"] = rxnorm_result.get("confidence", 0.0)
                else:
                    med_entry["rxnorm_code"] = None
                    med_entry["rxnorm_description"] = None
                    med_entry["confidence"] = 0.0
                extracted_medications.append(med_entry)

            return StructuredMedicalData(
                patient_info=patient_info,
                chief_complaint=chief_complaint,
                conditions=extracted_conditions,
                medications=extracted_medications,
                vital_signs=vital_signs,
                lab_results=lab_results,
                plan_items=plan_items,
                raw_note=note
            )
        except Exception as e:
            logger.error(f"Error extracting medical data: {str(e)}")
            raise
    
    async def batch_extract(self, notes: List[str], model: Optional[str] = None) -> List[StructuredMedicalData]:
        """
        Process multiple medical notes in parallel.
        
        Args:
            notes (List[str]): List of medical notes to process
            model (Optional[str]): Model to use for extraction
            
        Returns:
            List[StructuredMedicalData]: List of structured medical data
        """
        try:
            logger.info(f"Starting batch extraction of {len(notes)} notes")
            results = []
            for i, note in enumerate(notes, 1):
                logger.info(f"Processing note {i} of {len(notes)}")
                try:
                    result = await self.extract_medical_data(note, model=model)
                    results.append(result)
                    logger.info(f"Successfully processed note {i}")
                except Exception as e:
                    logger.error(f"Error processing note {i}: {str(e)}", exc_info=True)
                    raise
            logger.info(f"Completed batch extraction. Successfully processed {len(results)} notes")
            return results
        except Exception as e:
            logger.error(f"Error in batch extraction: {str(e)}", exc_info=True)
            raise 

    @staticmethod
    def clear_rxnorm_cache_for_med(med_name):
        """Utility to clear RxNorm cache for a medication (both lower and capitalized forms)."""
        import redis
        redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        redis_client.delete(f"rxnorm:{med_name.lower()}")
        redis_client.delete(f"rxnorm:{med_name.capitalize()}") 