from typing import List, Dict, Optional, Any
from openai import AsyncOpenAI
import json
import logging
from app.services.cache_service import CacheService, cache
from app.services.redis_service import redis_client
from app.services.rag.rag_service import RAGService
from app.services.rag.embedding.embedding_service import EmbeddingService
from app.services.rag.vector_db.chroma_service import ChromaService
from app.config import model_config
import re
from fastapi import HTTPException

# Add tiktoken import for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.logger = logging.getLogger(__name__)
        self.cache_service = CacheService()
        self.rag_service = RAGService()
        self.embedding_service = EmbeddingService()
        self.chroma_service = ChromaService()

    def _get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for the specified model."""
        model_name = model_name or model_config.DEFAULT_MODEL
        if model_name not in model_config.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported")
        return model_config.AVAILABLE_MODELS[model_name]

    def _count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in a string for a given model using tiktoken if available, else estimate."""
        if TIKTOKEN_AVAILABLE:
            try:
                enc = tiktoken.encoding_for_model(model_name)
                return len(enc.encode(text))
            except Exception:
                pass
        # Fallback: estimate 1 token per 4 chars
        return max(1, len(text) // 4)

    def _clean_json_response(self, response: str) -> str:
        """Remove triple backticks and optional 'json' language hint from LLM response."""
        cleaned = response.strip()
        # Remove triple backticks and optional 'json' after them
        if cleaned.startswith('```'):
            cleaned = re.sub(r'^```(?:json)?', '', cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        return cleaned

    @cache(expire=3600)
    async def generate_text(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """Generate text using the specified model, auto-adjusting max_tokens to fit context window."""
        try:
            model_cfg = self._get_model_config(model)
            model_name = model_cfg["name"]
            model_max_context = 8192 if "gpt-4" in model_name else 16385 if "gpt-3.5" in model_name else model_cfg.get("max_tokens", 4000)
            # Compose messages as in OpenAI API
            system_prompt = model_cfg["system_prompt"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            # Count tokens in all messages
            prompt_tokens = 0
            if TIKTOKEN_AVAILABLE:
                try:
                    enc = tiktoken.encoding_for_model(model_name)
                    for m in messages:
                        prompt_tokens += len(enc.encode(m["content"]))
                except Exception:
                    prompt_tokens = sum(self._count_tokens(m["content"], model_name) for m in messages)
            else:
                prompt_tokens = sum(self._count_tokens(m["content"], model_name) for m in messages)
            # Leave a buffer of 100 tokens
            max_tokens_allowed = model_max_context - prompt_tokens - 100
            if max_tokens is not None:
                max_tokens = min(max_tokens, max_tokens_allowed)
            else:
                max_tokens = min(model_cfg["max_tokens"], max_tokens_allowed)
            if max_tokens <= 0:
                raise ValueError(f"Prompt is too long for the model's context window.")
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=model_cfg["temperature"]
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise

    @cache(expire=3600)
    async def summarize_text(self, text: str, max_length: int = 200, model: Optional[str] = None) -> str:
        """Summarize medical text using the specified model."""
        try:
            prompt = f"""Please summarize the following medical note in {max_length} words or less.
            Focus on maintaining all critical medical information while being concise.
            Ensure the summary includes all important medical details, treatments, and recommendations.

            Medical Note:
            {text}"""
            return await self.generate_text(prompt, model=model, max_tokens=max_length)
        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}")
            raise

    @cache(expire=3600)
    async def paraphrase_text(self, text: str, model: Optional[str] = None) -> str:
        """Convert medical text into layman's terms while maintaining accuracy."""
        try:
            prompt = f"""Please convert the following medical note into simple, easy-to-understand language.
            Replace medical terminology with everyday language while maintaining the accuracy of the information.
            Keep all measurements, dates, and critical information intact.
            Make the text more accessible to non-medical readers.

            Guidelines:
            - Use simple, everyday language
            - Explain medical terms in plain English
            - Keep all important information and details
            - Maintain a professional but friendly tone
            - Ensure the meaning remains accurate

            Medical Note:
            {text}"""
            return await self.generate_text(prompt, model=model)
        except Exception as e:
            self.logger.error(f"Error paraphrasing text: {str(e)}")
            raise

    @cache(expire=3600)
    async def extract_entities(self, text: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Extract medical entities from text."""
        try:
            prompt = f"""You are a clinical information extraction assistant. Extract medical entities from the following text, follow these specific rules:

            CONDITION EXTRACTION RULES:
            1. Extract each medical condition as a SEPARATE, DISTINCT item. Only include actual medical diagnoses for Condition(e.g., diabetes, hypertension, meniscal tear). 
            Do NOT include symptoms or procedures here. Once you extract the conditions, unabbreviate them if they are abbreviated(eg. "Low HDL" -> "low high-density lipoprotein"), 
            and see if there is any analogous condition that is more common in the population. If there is, I want you to return the analogous condition in the "conditions" array(eg. "Low HDL" is equivalent to Lipoprotein deficiency, and Meniscal tear is equivalent to 
            Derangement of meniscus due to old tear or injury). If there is no analogous condition, return the condition as is.
            2. DO NOT combine multiple conditions into one entry (e.g., split "Overweight status, decreased HDL" into "Overweight" and "Low HDL")
            3. Remove descriptive words like "status", "condition", "disorder" - keep only the core medical term
            4. Use standard medical terminology (e.g., "Low HDL" instead of "decreased HDL")
            5. Do not include qualifiers like "possible", "probable", "mild", "severe" in the condition name
            6. Each condition should be a single, mappable medical diagnosis
            7. symptoms: Only include patient-reported symptoms or clinical findings (e.g., pain, stiffness, discomfort). Do NOT include diagnoses or procedures here.
            8. procedures: Only include medical or surgical procedures (e.g., arthroscopic meniscal repair, influenza immunization). Do NOT include diagnoses or symptoms here.
            9. medications: Only include medications (e.g., "Lisinopril", "Metformin", "Atorvastatin"). Do NOT include diagnoses or procedures here.
            
            - For each medication, extract the following fields:
            - "name": The medication name (e.g., "Atorvastatin")
            - "dosage": The dosage as written in the text (e.g., "20mg", "500 mg", "1 tablet"), or null if not present
            - "frequency": The frequency as written (e.g., "daily", "BID", "once a week"), or null if not present
            - "route": The route of administration (e.g., "oral", "IV"), or null if not present
            - "confidence": A float between 0 and 1 representing your confidence in the extraction
            - If any field is not present, set it to null.
            - Return the medications as a JSON array of objects, each with the fields: name, dosage, frequency, route, confidence.

            Special instructions:
            - For any phrase like "status-post [procedure]", extract the procedure under "procedures" and, if possible, infer and include the original condition that was treated under "conditions" (e.g., "status-post meniscal repair" → procedure: "meniscal repair", condition: "meniscal tear").
            - For ambiguous terms, use your best clinical judgment to classify as symptom, condition, or procedure.
            - For each condition, return the most standardized clinical name possible.
            - For each medication, extract the name, dosage (if present), frequency (if present), and route (if present).
            - If any of these fields are not present, set them to null.

            EXAMPLES:
            - Example: "Started Atorvastatin 20mg daily" → medications: [{{"name": "Atorvastatin", "dosage": "20mg", "frequency": "daily", "route": null, "confidence": 0.95}}]
            - Example: "Patient is on Metformin" → medications: [{{"name": "Metformin", "dosage": null, "frequency": null, "route": null, "confidence": 0.95}}]
            - "Overweight status, decreased HDL" → ["Overweight", "Low HDL"]
            - "Chronic kidney disease stage 3" → ["Chronic kidney disease"]
            - "Type 2 diabetes mellitus with complications" → ["Type 2 diabetes mellitus"]

            Return the result as a JSON object with the following structure:
            {{
                "patient_info": {{
                    "name": "patient name",
                    "age": "age",
                    "gender": "gender",
                    "mrn": "medical record number"
                }},
                "chief_complaint": "main reason for visit",
                "conditions": ["Overweight", "Low High-Density Lipoprotein", "Meniscal tear"],
                "symptoms": ["Stiffness", "Discomfort"],
                "procedures": ["Meniscal repair", "Arthroscopic meniscal repair"],
                "condition_status": {{
                    "Overweight": "active",
                    "Low High-Density Lipoprotein": "active"
                }},
                "condition_onset": {{
                    "Overweight": "onset date/time",
                    "Low High-Density Lipoprotein": "onset date/time"
                }},
                "condition_confidence": {{
                    "Overweight": 0.95,
                    "Low High-Density Lipoprotein": 0.95
                }},
                "medications": [
                    {{"name":"Atorvastatin", "dosage": "20mg", "frequency": "daily", "route": "oral", "confidence": 0.95}},
                    {{"name": "Metformin", "dosage": null, "frequency": null, "route": null, "confidence": 0.95}}
                ],
                "vital_signs": [
                    {{"name": "BP", "value": "140/90", "unit": "mmHg", "normal_range": "120/80-140/90"}},
                    {{"name": "HR", "value": "88", "unit": "bpm", "normal_range": "60-100"}}
                ],
                "lab_results": [
                    {{"test_name": "test name", "value": "result", "unit": "unit", "reference_range": "range", "status": "normal/abnormal/critical"}}
                ],
                "plan_items": [
                    {{"action": "action", "category": "medication/follow_up/test/referral/lifestyle", "details": "details"}}
                ]
            }}

            Medical Note:
            {text}"""
            response = await self.generate_text(prompt, model=model)
            response = self._clean_json_response(response)
            parsed = json.loads(response)
            if not isinstance(parsed, dict):
                raise ValueError(f"LLM output is not a JSON object: {parsed}")
            return parsed
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            raise

    @cache(expire=3600)
    async def classify_text(self, text: str, categories: List[str]) -> str:
        """Classify text into one of the given categories."""
        try:
            prompt = f"Classify the following text into one of these categories: {', '.join(categories)}\n\nText: {text}\n\nReturn only the category name."
            return await self.generate_text(prompt)
        except Exception as e:
            self.logger.error(f"Error classifying text: {str(e)}")
            raise

    @cache(expire=3600)
    async def process_note(self, note_text: str, tasks: List[str], model: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a medical note using a single API call to handle all tasks.
        
        Args:
            note_text (str): The medical note text to process
            tasks (List[str]): List of tasks to perform (e.g., ["summarize", "paraphrase", "extract_entities"])
            model (Optional[str]): Override the default model for all tasks
            
        Returns:
            Dict[str, Any]: Dictionary mapping task names to their results
        """
        try:
            # Input validation
            if not note_text or not isinstance(note_text, str) or not note_text.strip():
                raise HTTPException(status_code=400, detail="Input note is empty or invalid.")
            model_config = self._get_model_config(model)
            
            # Create a single prompt that handles all tasks
            prompt = f"""Please process the following medical note and provide the following outputs:

1. Summary: Provide a concise summary of the medical note, focusing on key medical information, symptoms, and treatment plan.
2. Paraphrase: Convert the medical note into simple, everyday language while maintaining accuracy.
3. Medical Information: Extract and categorize medical information in JSON format, including:
   - Vital signs (temperature, blood pressure, heart rate, respiratory rate)
   - Symptoms
   - Physical examination findings
   - Assessment/Diagnosis
   - Treatment plan/Recommendations

Medical Note:
{note_text}

Please provide the output in the following JSON format:
{{
    "summary": "concise summary here",
    "paraphrase": "simple language version here",
    "medical_info": {{
        "vital_signs": {{
            "temperature": "value",
            "blood_pressure": "value",
            "heart_rate": "value",
            "respiratory_rate": "value"
        }},
        "symptoms": ["symptom1", "symptom2", ...],
        "physical_exam": ["finding1", "finding2", ...],
        "assessment": ["diagnosis1", "diagnosis2", ...],
        "treatment": ["treatment1", "treatment2", ...]
    }}
}}"""

            # Make a single API call
            response = await self.client.chat.completions.create(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": model_config["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"]
            )
            
            # Parse the response
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise HTTPException(status_code=500, detail="LLM returned an empty response.")
            result = json.loads(content)
            
            # Map the results to the requested tasks
            results = {}
            if "summarize" in tasks:
                results["summarize"] = result["summary"]
            if "paraphrase" in tasks:
                results["paraphrase"] = result["paraphrase"]
            if "extract_entities" in tasks:
                results["extract_entities"] = result["medical_info"]
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing note: {str(e)}")
            raise 

def normalize_medication_dict(med):
    # Ensure all required fields are present
    return {
        "name": med.get("name"),
        "dosage": med.get("dosage"),
        "frequency": med.get("frequency"),
        "route": med.get("route"),
        "confidence": float(med.get("confidence", 1.0)),
        "rxnorm_code": med.get("rxnorm_code"),
        "rxnorm_description": med.get("rxnorm_description"),
    } 