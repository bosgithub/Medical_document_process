from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ExtractedPatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    mrn: Optional[str] = None

class ExtractedCondition(BaseModel):
    name: str
    icd_code: Optional[str] = None
    icd_description: Optional[str] = None
    status: Optional[str] = None  # active, resolved, chronic
    onset: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ExtractedMedication(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rxnorm_code: Optional[str] = None
    rxnorm_description: Optional[str] = None

class ExtractedVitalSign(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    normal_range: Optional[str] = None

class ExtractedLabResult(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    status: Optional[str] = None  # normal, abnormal, critical

class ExtractedPlan(BaseModel):
    action: str
    category: str  # medication, follow_up, test, referral, lifestyle
    details: Optional[str] = None

class StructuredMedicalData(BaseModel):
    patient_info: ExtractedPatientInfo
    chief_complaint: Optional[str] = None
    conditions: List[ExtractedCondition] = []
    medications: List[ExtractedMedication] = []
    vital_signs: List[ExtractedVitalSign] = []
    lab_results: List[ExtractedLabResult] = []
    plan_items: List[ExtractedPlan] = []
    raw_note: str
    extraction_timestamp: datetime = Field(default_factory=datetime.now) 