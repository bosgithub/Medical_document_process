from fastapi import FastAPI, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import json
from datetime import datetime
import re
import os

from app import crud, models, schemas
from app.database import SessionLocal, engine
from app.create_tables import create_tables
from app.services.llm.llm_service import LLMService
from app.services.rag.rag_service import RAGService
from app.services.cache_service import CacheService
from app.services.redis_service import redis_client
from app.services.rag.embedding.embedding_service import EmbeddingService
from app.services.rag.vector_db.chroma_service import ChromaService
from app.services.extraction.extraction_service import ExtractionService
from app.services.extraction.extraction_schemas import StructuredMedicalData
from pydantic import BaseModel, Field
from fhir.resources.patient import Patient as FhirPatient
from fhir.resources.condition import Condition as FhirCondition
from fhir.resources.medicationrequest import MedicationRequest as FhirMedicationRequest
from fhir.resources.observation import Observation as FhirObservation
from fhir.resources.encounter import Encounter as FhirEncounter
from fhir.resources.careplan import CarePlan as FhirCarePlan
from fhir.resources.reference import Reference
from fhir.resources.humanname import HumanName
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.period import Period
from fhir.resources.dosage import Dosage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create database tables
create_tables()

app = FastAPI(
    title="Medical Documents API",
    description="""
    API for processing and managing medical documents. This service provides:
    
    * Medical note extraction with structured data
    * ICD-10 code lookup for conditions
    * Document storage and retrieval
    * Question answering using RAG
    
    ## Key Features
    
    * **Structured Data Extraction**: Convert unstructured medical notes into structured data
    * **Code Lookup**: Automatically find relevant medical codes
    * **Document Management**: Store and retrieve medical documents
    * **Question Answering**: Get answers from medical documents using RAG
    
    ## Authentication
    
    This API requires authentication. Please contact the administrator for API keys.
    
    ## Rate Limiting
    
    * 100 requests per minute per API key
    * 1000 requests per hour per API key
    
    ## Error Handling
    
    The API uses standard HTTP status codes:
    * 200: Success
    * 400: Bad Request
    * 401: Unauthorized
    * 403: Forbidden
    * 404: Not Found
    * 429: Too Many Requests
    * 500: Internal Server Error
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Initialize services
llm_service = LLMService()
rag_service = RAGService()
extraction_service = ExtractionService()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check endpoint
@app.get("/health", 
    summary="Health Check",
    description="Check if the API is running and healthy.",
    response_description="Returns the status of the API.",
    tags=["System"]
)
def health_check():
    return {"status": "ok"}

# Document endpoints
@app.get("/documents/", response_model=List[schemas.Document])
def read_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    documents = crud.get_documents(db, skip=skip, limit=limit)
    return documents

@app.get("/documents/{document_id}", response_model=schemas.Document)
def read_document(document_id: int, db: Session = Depends(get_db)):
    db_document = crud.get_document(db, document_id=document_id)
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document

@app.post("/documents/", response_model=schemas.Document)
def create_document(document: schemas.DocumentCreate, db: Session = Depends(get_db)):
    try:
        logger.info(f"Creating document with title: {document.title}")
        return crud.create_document(db=db, document=document)
    except Exception as e:
        logger.error(f"Error creating document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/documents/{document_id}", response_model=schemas.Document)
def update_document(document_id: int, document: schemas.DocumentCreate, db: Session = Depends(get_db)):
    db_document = crud.update_document(db, document_id=document_id, document=document)
    if db_document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_document

@app.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    success = crud.delete_document(db, document_id=document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success"}

# AI Processing endpoints
@app.post("/process_note", response_model=schemas.ProcessNoteResponse)
async def process_note_endpoint(
    request: schemas.ProcessNoteRequest,
    model: Optional[str] = None
):
    """
    Process a medical note using LLM.
    
    Args:
        request (ProcessNoteRequest): The request containing the note text and tasks
        model (str, optional): Override the default model for all tasks
        
    Returns:
        ProcessNoteResponse: The processed note results for each task
    """
    try:
        results = await llm_service.process_note(
            note_text=request.note_text,
            tasks=request.tasks,
            model=model
        )
        return schemas.ProcessNoteResponse(results=results)
    except Exception as e:
        logger.error(f"Error in process_note endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/answer_question", response_model=schemas.QuestionResponse)
async def answer_question(
    question: schemas.QuestionRequest,
    db: Session = Depends(get_db)
):
    """
    Answer a question using RAG on the available documents.
    
    Args:
        question: The question to answer
        db: Database session
        
    Returns:
        QuestionResponse containing the answer and source documents
    """
    try:
        # Get all documents from the database
        documents = crud.get_documents(db)
        
        # Convert SQLAlchemy models to dicts
        doc_dicts = [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content
            }
            for doc in documents
        ]
        
        # Get answer using RAG
        result = await rag_service.answer_question(
            question=question.question,
            documents=doc_dicts
        )
        
        return schemas.QuestionResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error in answer_question endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

# Medical Data Extraction endpoint
class ExtractStructuredRequest(BaseModel):
    """Request model for /extract_structured endpoint"""
    note_text: str = Field(
        ...,
        description="The medical note text to process",
        example="""
        Patient: John Doe, 45M
        Chief Complaint: Chest pain and shortness of breath

        Vital Signs:
        BP: 140/90 mmHg
        HR: 88 bpm
        Temp: 98.6 F
        RR: 18

        Assessment:
        1. Hypertension (ICD-10: I10)
        2. Type 2 Diabetes (ICD-10: E11.9)
        3. Hyperlipidemia (ICD-10: E78.5)

        Medications:
        1. Lisinopril 10mg daily (RxNorm: 197319)
        2. Metformin 500mg BID (RxNorm: 6809)
        3. Atorvastatin 20mg daily (RxNorm: 83367)

        Plan:
        1. Follow up in 2 weeks
        2. Check fasting lipid panel
        3. Continue current medications
        """
    )

    class Config:
        json_schema_extra = {
            "example": {
                "note_text": """
                Patient: John Doe, 45M
                Chief Complaint: Chest pain and shortness of breath

                Vital Signs:
                BP: 140/90 mmHg
                HR: 88 bpm
                Temp: 98.6 F
                RR: 18

                Assessment:
                1. Hypertension (ICD-10: I10)
                2. Type 2 Diabetes (ICD-10: E11.9)
                3. Hyperlipidemia (ICD-10: E78.5)

                Medications:
                1. Lisinopril 10mg daily (RxNorm: 197319)
                2. Metformin 500mg BID (RxNorm: 6809)
                3. Atorvastatin 20mg daily (RxNorm: 83367)

                Plan:
                1. Follow up in 2 weeks
                2. Check fasting lipid panel
                3. Continue current medications
                """
            }
        }

@app.post("/extract_structured", 
    response_model=StructuredMedicalData,
    summary="Extract Structured Medical Data",
    description="""
    Extract structured medical data from a medical note. This endpoint:
    
    * Processes the medical note text
    * Extracts patient information
    * Identifies conditions and looks up ICD-10 codes
    * Extracts vital signs and lab results
    * Identifies plan items
    
    The response includes all extracted information in a structured format.
    """,
    response_description="Structured medical data including patient info, conditions, etc.",
    tags=["Extraction"]
)
async def extract_structured(request: ExtractStructuredRequest):
    """
    Extract structured medical data from a note.
    
    Args:
        request (ExtractStructuredRequest): The request containing the medical note
        
    Returns:
        StructuredMedicalData: Structured medical information including:
            - Patient information
            - Conditions with ICD codes
            - Vital signs
            - Lab results
            - Plan items
            
    Raises:
        HTTPException: If there's an error processing the note
    """
    try:
        logger.info("Received request to extract structured data")
        logger.debug(f"Note length: {len(request.note_text)} characters")
        
        start_time = datetime.now()
        result = await extraction_service.extract_medical_data(request.note_text)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Successfully extracted data in {processing_time:.2f} seconds")
        logger.debug(f"Extracted data summary: {len(result.conditions)} conditions, "
                    f"{len(result.medications)} medications")
        
        return result
    except Exception as e:
        logger.error(f"Error in extract_structured endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing medical note: {str(e)}"
        )

@app.post("/extract/batch", response_model=List[StructuredMedicalData])
async def batch_extract_medical_data(notes: List[str]):
    """
    Process multiple medical notes in parallel.
    
    Args:
        notes (List[str]): List of medical notes to process
        
    Returns:
        List[StructuredMedicalData]: List of structured medical data
    """
    try:
        results = await extraction_service.batch_extract(notes)
        return results
    except Exception as e:
        logger.error(f"Error in batch_extract_medical_data endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing medical notes: {str(e)}"
        )

@app.post("/documents/{document_id}/extract", response_model=StructuredMedicalData)
async def extract_from_document(document_id: int, db: Session = Depends(get_db)):
    """
    Extract medical data from a stored document.
    
    Args:
        document_id (int): ID of the document to process
        db (Session): Database session
        
    Returns:
        StructuredMedicalData: Structured medical information
    """
    try:
        # Get document from database
        document = crud.get_document(db, document_id=document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Extract medical data
        result = await extraction_service.extract_medical_data(document.content)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_from_document endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

def to_fhir_bundle(structured_data: StructuredMedicalData) -> dict:
    """
    Map StructuredMedicalData to a simplified FHIR bundle containing Patient, Condition, MedicationRequest, Observation, CarePlan, and Encounter resources.
    Procedure resources are not included.
    """
    patient_id = structured_data.patient_info.mrn or "001"
    # Extract encounter date from raw_note if possible (very basic extraction)
    encounter_date = None
    if structured_data.raw_note:
        match = re.search(r'Encounter Date[:\-]?\s*([\d\-/]+)', structured_data.raw_note)
        if match:
            encounter_date = match.group(1)
    encounter_id = f"enc-{patient_id}"
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "name": structured_data.patient_info.name,
        "gender": structured_data.patient_info.gender,
    }
    # Encounter resource (one per note)
    encounter_resource = {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
        "class": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB"}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "period": {"start": encounter_date} if encounter_date else {},
    }
    # Condition resources
    condition_resources = []
    for cond in structured_data.conditions:
        condition_resources.append({
            "resourceType": "Condition",
            "code": {
                "coding": [
                    {"system": "http://hl7.org/fhir/sid/icd-10", "code": cond.icd_code, "display": cond.icd_description}
                ],
                "text": cond.name
            },
            "clinicalStatus": "active",
            "verificationStatus": "confirmed",
            "subject": {"reference": f"Patient/{patient_id}"},
            "onsetString": cond.onset or None,
            "confidence": cond.confidence
        })
    # MedicationRequest resources
    medication_resources = []
    for med in structured_data.medications:
        medication_resources.append({
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {
                "coding": [
                    {"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": med.rxnorm_code, "display": med.rxnorm_description}
                ],
                "text": med.name
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "dosageInstruction": [{
                "text": f"{med.dosage or ''} {med.frequency or ''} {med.route or ''}".strip()
            }],
            "confidence": med.confidence
        })
    # Observation resources for vital signs and lab results
    observation_resources = []
    for vital in structured_data.vital_signs:
        observation_resources.append({
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"text": vital.name},
            "subject": {"reference": f"Patient/{patient_id}"},
            "valueString": vital.value,
            "unit": vital.unit,
            "normalRange": vital.normal_range
        })
    for lab in structured_data.lab_results:
        obs = FhirObservation.construct(
            status="final",
            category=[CodeableConcept.construct(coding=[Coding.construct(system="http://terminology.hl7.org/CodeSystem/observation-category", code="laboratory")])],
            code=CodeableConcept.construct(text=lab.test_name),
            subject=Reference.construct(reference=f"Patient/{patient_id}"),
            valueString=lab.value
        )
        observation_resources.append(obs)
    # CarePlan resources for plan items
    careplan_resources = []
    for plan in structured_data.plan_items:
        careplan_resources.append({
            "resourceType": "CarePlan",
            "status": "active",
            "intent": "plan",
            "title": plan.action,
            "category": [{"text": plan.category}],
            "description": plan.details,
            "subject": {"reference": f"Patient/{patient_id}"}
        })
    # Return FHIR bundle in the specified order
    return {
        "patient": patient_resource,
        "conditions": condition_resources,
        "medications": medication_resources,
        "observations": observation_resources,
        "encounter": encounter_resource,
        "careplans": careplan_resources
    }

@app.post("/to_fhir", summary="Convert structured data to FHIR-like JSON", tags=["FHIR"])
async def to_fhir_endpoint(
    structured_data: StructuredMedicalData = Body(..., example={
        "patient_info": {"name": "John Doe", "age": "45", "gender": "male", "mrn": "123"},
        "chief_complaint": "Chest pain",
        "conditions": [{"name": "Hypertension", "icd_code": "I10", "icd_description": "Hypertension", "confidence": 0.95}],
        "medications": [{"name": "Lisinopril", "dosage": "10mg", "frequency": "daily", "route": "oral", "confidence": 0.95, "rxnorm_code": "197319", "rxnorm_description": "Lisinopril"}],
        "vital_signs": [],
        "lab_results": [],
        "plan_items": [],
        "raw_note": "..."
    })
):
    """
    Convert structured medical data to a FHIR-like JSON bundle.
    """
    fhir_bundle = to_fhir_bundle(structured_data)
    return fhir_bundle

# Strict FHIR mapping using fhir.resources
def to_fhir_strict_resources(structured_data: StructuredMedicalData):
    """
    Map StructuredMedicalData to a list of real FHIR resource objects using fhir.resources.
    """
    resources = []
    # Patient
    patient_id = structured_data.patient_info.mrn or "001"
    patient = FhirPatient.construct(
        id=patient_id,
        name=[HumanName.construct(text=structured_data.patient_info.name)],
        gender=structured_data.patient_info.gender
    )
    resources.append(patient)
    # Encounter
    import re
    encounter_date = None
    if structured_data.raw_note:
        match = re.search(r'Encounter Date[:\-]?\s*([\d\-/]+)', structured_data.raw_note)
        if match:
            encounter_date = match.group(1)
    encounter = FhirEncounter.construct(
        id=f"enc-{patient_id}",
        status="finished",
        class_fhir=Coding.construct(system="http://terminology.hl7.org/CodeSystem/v3-ActCode", code="AMB"),
        subject=Reference.construct(reference=f"Patient/{patient_id}"),
        period=Period.construct(start=encounter_date) if encounter_date else None
    )
    resources.append(encounter)
    # Conditions
    for cond in structured_data.conditions:
        condition = FhirCondition.construct(
            code=CodeableConcept.construct(
                coding=[Coding.construct(system="http://hl7.org/fhir/sid/icd-10", code=cond.icd_code, display=cond.icd_description)],
                text=cond.name
            ),
            clinicalStatus=CodeableConcept.construct(text="active"),
            verificationStatus=CodeableConcept.construct(text="confirmed"),
            subject=Reference.construct(reference=f"Patient/{patient_id}"),
            onsetString=cond.onset or None
        )
        resources.append(condition)
    # Medications
    for med in structured_data.medications:
        med_req = FhirMedicationRequest.construct(
            medicationCodeableConcept=CodeableConcept.construct(
                coding=[Coding.construct(system="http://www.nlm.nih.gov/research/umls/rxnorm", code=med.rxnorm_code, display=med.rxnorm_description)],
                text=med.name
            ),
            subject=Reference.construct(reference=f"Patient/{patient_id}"),
            dosageInstruction=[Dosage.construct(text=f"{med.dosage or ''} {med.frequency or ''} {med.route or ''}".strip())]
        )
        resources.append(med_req)
    # Observations (vitals and labs)
    for vital in structured_data.vital_signs:
        obs = FhirObservation.construct(
            status="final",
            category=[CodeableConcept.construct(coding=[Coding.construct(system="http://terminology.hl7.org/CodeSystem/observation-category", code="vital-signs")])],
            code=CodeableConcept.construct(text=vital.name),
            subject=Reference.construct(reference=f"Patient/{patient_id}"),
            valueString=vital.value
        )
        resources.append(obs)
    for lab in structured_data.lab_results:
        obs = FhirObservation.construct(
            status="final",
            category=[CodeableConcept.construct(coding=[Coding.construct(system="http://terminology.hl7.org/CodeSystem/observation-category", code="laboratory")])],
            code=CodeableConcept.construct(text=lab.test_name),
            subject=Reference.construct(reference=f"Patient/{patient_id}"),
            valueString=lab.value
        )
        resources.append(obs)
    # CarePlans
    for plan in structured_data.plan_items:
        careplan = FhirCarePlan.construct(
            status="active",
            intent="plan",
            title=plan.action,
            category=[CodeableConcept.construct(text=plan.category)],
            description=plan.details,
            subject=Reference.construct(reference=f"Patient/{patient_id}")
        )
        resources.append(careplan)
    return resources

@app.post("/to_fhir_strict", summary="Convert structured data to strict FHIR resources", tags=["FHIR"])
async def to_fhir_strict_endpoint(
    structured_data: StructuredMedicalData = Body(..., example={
        "patient_info": {"name": "John Doe", "age": "45", "gender": "male", "mrn": "123"},
        "chief_complaint": "Chest pain",
        "conditions": [{"name": "Hypertension", "icd_code": "I10", "icd_description": "Hypertension", "confidence": 0.95}],
        "medications": [{"name": "Lisinopril", "dosage": "10mg", "frequency": "daily", "route": "oral", "confidence": 0.95, "rxnorm_code": "197319", "rxnorm_description": "Lisinopril"}],
        "vital_signs": [],
        "lab_results": [],
        "plan_items": [],
        "raw_note": "..."
    })
):
    """
    Convert structured medical data to a list of strict FHIR resource objects (serialized to JSON).
    """
    resources = to_fhir_strict_resources(structured_data)
    # Serialize each resource to dict
    return [r.dict() for r in resources]

@app.on_event("startup")
def seed_database_with_soap_note():
    from app import crud, schemas
    from app.database import SessionLocal
    db = SessionLocal()
    # Only seed if the documents table is empty
    if not crud.get_documents(db):
        soap_note_path = os.path.join(os.path.dirname(__file__), "soap_notes", "soap_01.txt")
        if os.path.exists(soap_note_path):
            with open(soap_note_path, "r") as f:
                note_content = f.read()
            doc = schemas.DocumentCreate(
                title="Sample SOAP Note 01",
                content=note_content
            )
            crud.create_document(db, document=doc)
            print("Seeded database with SOAP Note 01.")
        else:
            print(f"SOAP note not found at {soap_note_path}, skipping seed.")
    db.close()

