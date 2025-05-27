from typing import Dict, Any
from pydantic_settings import BaseSettings

class ModelConfig(BaseSettings):
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
        "gpt-3.5-turbo": {
            "name": "gpt-3.5-turbo",
            "max_tokens": 4000,
            "temperature": 0.0,
            "provider": "openai",
            "cost_per_1k_tokens": 0.002,
            "system_prompt": """You are a medical document processing assistant. \n            Focus on accuracy and completeness when processing medical notes.\n            When paraphrasing, convert medical terminology into simple, everyday language.\n            Maintain all critical medical information, including:\n            - Patient symptoms and conditions (in simple terms)\n            - Medical procedures and treatments (explained clearly)\n            - Medications and dosages\n            - Test results and measurements\n            - Follow-up recommendations (in plain language)"""
        },
        "gpt-3.5-turbo-zero-temp": {
            "name": "gpt-3.5-turbo",
            "max_tokens": 4000,
            "temperature": 0.0,
            "provider": "openai",
            "cost_per_1k_tokens": 0.002,
            "system_prompt": """You are a medical document processing assistant. \n            Focus on accuracy and completeness when extracting entities from medical notes.\n            Output must be deterministic and consistent for the same input."""
        },
        "gpt-4": {
            "name": "gpt-4",
            "max_tokens": 8000,
            "temperature": 0.0,
            "provider": "openai",
            "cost_per_1k_tokens": 0.03,
            "system_prompt": """You are a medical document processing assistant. \n            Focus on accuracy and completeness when processing medical notes.\n            When paraphrasing, convert medical terminology into simple, everyday language.\n            Maintain all critical medical information, including:\n            - Patient symptoms and conditions (in simple terms)\n            - Medical procedures and treatments (explained clearly)\n            - Medications and dosages\n            - Test results and measurements\n            - Follow-up recommendations (in plain language)\n            - Risk factors and contraindications (explained simply)\n            - Medical history and family history (in accessible terms)"""
        },
        "claude-3-opus": {
            "name": "claude-3-opus",
            "max_tokens": 8000,
            "temperature": 0.0,
            "provider": "anthropic",
            "cost_per_1k_tokens": 0.015,
            "system_prompt": """You are a medical document processing assistant. \n            Focus on accuracy and completeness when processing medical notes.\n            When paraphrasing, convert medical terminology into simple, everyday language.\n            Maintain all critical medical information, including:\n            - Patient symptoms and conditions (in simple terms)\n            - Medical procedures and treatments (explained clearly)\n            - Medications and dosages\n            - Test results and measurements\n            - Follow-up recommendations (in plain language)\n            - Risk factors and contraindications (explained simply)\n            - Medical history and family history (in accessible terms)"""
        }
    }

    class Config:
        env_prefix = "LLM_"

model_config = ModelConfig() 