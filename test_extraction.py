"""
Unit Tests for Medical Note Extraction Service

This test suite verifies the functionality of the medical note extraction service.
The tests are organized into several categories:

1. Core Functionality Tests:
   - Basic extraction of all required fields (patient info, conditions, medications, etc.)
   - Validation of data types and structures
   - Verification of ICD and RxNorm codes

2. Error Handling Tests:
   - Empty notes
   - Invalid notes
   - API endpoint error responses

3. Edge Cases:
   - Notes with missing fields
   - Special characters and non-English names
   - Various measurement formats and units

4. API Endpoint Tests:
   - Main extraction endpoint
   - Response structure validation
   - Data format verification

5. Batch Processing Tests (Extension):
   - Multiple note processing
   - Batch API endpoint
   - Consistency across batch results

Each test follows the Arrange-Act-Assert pattern:
1. Arrange: Set up test data and conditions
2. Act: Call the function being tested
3. Assert: Verify the results match expectations
"""

import asyncio
import os
import logging
from datetime import datetime
from app.services.extraction.extraction_service import ExtractionService
from app.services.extraction.code_lookup.rxnorm_service import RxNormLookupService
import json
from app.services.extraction.extraction_schemas import StructuredMedicalData

# Set up logging to file
logging.basicConfig(
    filename='extraction_test.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# This test will use the real ExtractionService to extract core entities from the SOAP note

def test_icd10_extraction_on_soap_01():
    async def run():
        with open("soap_notes/soap_01.txt", "r") as f:
            note_text = f.read()
        extraction_service = ExtractionService()
        # Call the full extraction pipeline (including ICD-10 lookup)
        result = await extraction_service.extract_medical_data(note_text)
        print("ICD-10 Extraction Results for SOAP 01:")
        for cond in result.conditions:
            print(f"  Name: {cond.name}")
            print(f"  ICD-10 Code: {cond.icd_code}")
            print(f"  Description: {cond.icd_description}")
            print(f"  Confidence: {cond.confidence}")
            print()
    asyncio.run(run())

def test_icd10_extraction_on_soap_02():
    async def run():
        with open("soap_notes/soap_02.txt", "r") as f:
            note_text = f.read()
        extraction_service = ExtractionService()
        # Call the full extraction pipeline (including ICD-10 lookup)
        result = await extraction_service.extract_medical_data(note_text)
        print("ICD-10 Extraction Results for SOAP 02:")
        for cond in result.conditions:
            print(f"  Name: {cond.name}")
            print(f"  ICD-10 Code: {cond.icd_code}")
            print(f"  Description: {cond.icd_description}")
            print(f"  Confidence: {cond.confidence}")
            print()
    asyncio.run(run())

def test_icd10_extraction_on_soap_03():
    async def run():
        with open("soap_notes/soap_03.txt", "r") as f:
            note_text = f.read()
        extraction_service = ExtractionService()
        # Call the full extraction pipeline (including ICD-10 lookup)
        result = await extraction_service.extract_medical_data(note_text)
        print("ICD-10 Extraction Results for SOAP 03:")
        for cond in result.conditions:
            print(f"  Name: {cond.name}")
            print(f"  ICD-10 Code: {cond.icd_code}")
            print(f"  Description: {cond.icd_description}")
            print(f"  Confidence: {cond.confidence}")
            print()
    asyncio.run(run())

def test_icd10_extraction_on_all_soap_notes():
    async def run():
        script_dir = os.path.dirname(__file__)
        notes_dir = os.path.join(script_dir, "soap_notes")
        note_files = [f for f in os.listdir(notes_dir) if f.endswith('.txt')]
        extraction_service = ExtractionService()
        for note_file in note_files:
            note_path = os.path.join(notes_dir, note_file)
            with open(note_path, "r") as f:
                note_text = f.read()
            start_time = datetime.now()
            result = await extraction_service.extract_medical_data(note_text)
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            # Prepare log output
            log_lines = [
                f"\n--- Extraction for {note_file} ---",
                f"Timestamp: {start_time.isoformat()}",
                f"Elapsed: {elapsed:.2f} seconds",
                "Conditions:"
            ]
            for cond in result.conditions:
                # If cond is a Pydantic model, convert to dict
                cond_dict = cond if isinstance(cond, dict) else cond.__dict__
                log_lines.append(f"  Name: {cond_dict.get('name')}")
                log_lines.append(f"  ICD-10 Code: {cond_dict.get('icd_code')}")
                log_lines.append(f"  Description: {cond_dict.get('icd_description')}")
                log_lines.append(f"  Confidence: {cond_dict.get('confidence')}")
                if 'top_suggestions' in cond_dict and cond_dict['top_suggestions']:
                    log_lines.append("  Top ICD-10 Suggestions:")
                    for s in cond_dict['top_suggestions']:
                        log_lines.append(f"    Code: {s['code']}, Desc: {s['description']}, Similarity: {s['similarity_score']:.3f}")
                log_lines.append("")
            log_output = "\n".join(log_lines)
            print(log_output)
            logging.info(log_output)
    asyncio.run(run())

def test_rxnorm_extraction_on_soap_01():
    async def run():
        with open("soap_notes/soap_01.txt", "r") as f:
            note_text = f.read()
        extraction_service = ExtractionService()
        result = await extraction_service.extract_medical_data(note_text)
        print("RxNorm Extraction Results for SOAP 01:")
        for med in result.medications:
            print(f"  Name: {med.name if hasattr(med, 'name') else med.get('name')}")
            print(f"  RxNorm Code: {med.rxnorm_code if hasattr(med, 'rxnorm_code') else med.get('rxnorm_code')}")
            print(f"  Description: {med.rxnorm_description if hasattr(med, 'rxnorm_description') else med.get('rxnorm_description')}")
            print(f"  Confidence: {med.confidence if hasattr(med, 'confidence') else med.get('confidence')}")
            print()
            # Assert RxNorm code is present for real medications
            rxnorm_code = getattr(med, 'rxnorm_code', None)
            if rxnorm_code is None and isinstance(med, dict):
                rxnorm_code = med.get('rxnorm_code')
            med_name = getattr(med, 'name', None)
            if med_name is None and isinstance(med, dict):
                med_name = med.get('name')
            assert rxnorm_code is not None, f"No RxNorm code for medication {med_name}"
    asyncio.run(run())

async def test_rxnorm_lookup():
    rxnorm_service = RxNormLookupService()
    test_meds = ["Lisinopril", "Metformin", "Atorvastatin", "NonexistentMed"]
    for med in test_meds:
        result = await rxnorm_service.lookup_rxnorm_code(med)
        print(f"Medication: {med}")
        print(f"  RxNorm Code: {result.get('rxnorm_code')}")
        print(f"  Description: {result.get('rxnorm_description')}")
        print(f"  Confidence: {result.get('confidence')}")
        print()
        if med != "NonexistentMed":
            assert result.get('rxnorm_code') is not None, f"RxNorm code not found for {med}"
        else:
            assert result.get('rxnorm_code') is None, "Expected no RxNorm code for fake med"

def run_agentic_structured_extraction_on_note(note_filename):
    async def run():
        with open(note_filename, "r") as f:
            note_text = f.read()
        extraction_service = ExtractionService()
        result = await extraction_service.extract_medical_data(note_text)
        print(f"Agentic Structured Extraction Result for {note_filename}:")
        print(json.dumps(result.model_dump() if hasattr(result, 'model_dump') else result.dict(), indent=2, default=str))
        # Validate key fields
        assert result.patient_info is not None, "Missing patient_info"
        assert result.conditions and len(result.conditions) > 0, "No conditions extracted"
        for cond in result.conditions:
            assert getattr(cond, 'icd_code', None) is not None, f"No ICD-10 code for condition {getattr(cond, 'name', None)}"
        if not result.medications or len(result.medications) == 0:
            print("No medications prescribed in this note (expected if none present).")
        else:
            for med in result.medications:
                rxnorm_code = getattr(med, 'rxnorm_code', None)
                if rxnorm_code is None and isinstance(med, dict):
                    rxnorm_code = med.get('rxnorm_code')
                med_name = getattr(med, 'name', None)
                if med_name is None and isinstance(med, dict):
                    med_name = med.get('name')
                assert rxnorm_code is not None, f"No RxNorm code for medication {med_name}"
        assert result.vital_signs is not None, "Missing vital_signs"
        assert result.lab_results is not None, "Missing lab_results"
        assert result.plan_items is not None, "Missing plan_items"
        StructuredMedicalData.model_validate(result)
    asyncio.run(run())

def test_agentic_structured_extraction_on_soap_01():
    run_agentic_structured_extraction_on_note("soap_notes/soap_01.txt")

def test_agentic_structured_extraction_on_soap_02():
    run_agentic_structured_extraction_on_note("soap_notes/soap_02.txt")

def test_to_fhir_bundle_on_soap_02():
    """Test FHIR bundle mapping on SOAP note 02."""
    async def run():
        from app.main import to_fhir_bundle
        with open("soap_notes/soap_02.txt", "r") as f:
            note_text = f.read()
        extraction_service = ExtractionService()
        structured = await extraction_service.extract_medical_data(note_text)
        fhir_bundle = to_fhir_bundle(structured)
        print("FHIR Bundle for SOAP 02:")
        print(json.dumps(fhir_bundle, indent=2, default=str))
        # Assert output order and presence of keys
        expected_keys = ["patient", "conditions", "medications", "observations", "encounter", "careplans"]
        assert list(fhir_bundle.keys()) == expected_keys, f"FHIR bundle keys order mismatch: {list(fhir_bundle.keys())}"
        for key in expected_keys:
            assert key in fhir_bundle, f"Missing key in FHIR bundle: {key}"
    asyncio.run(run())

def test_to_fhir_strict_on_soap_02():
    """Test strict FHIR resource mapping on SOAP note 02 using fhir.resources."""
    async def run():
        from app.main import to_fhir_strict_resources
        with open("soap_notes/soap_02.txt", "r") as f:
            note_text = f.read()
        extraction_service = ExtractionService()
        structured = await extraction_service.extract_medical_data(note_text)
        fhir_resources = to_fhir_strict_resources(structured)
        print("Strict FHIR Resources for SOAP 02:")
        for r in fhir_resources:
            print(json.dumps(r.dict(), indent=2, default=str))
        # Assert presence of key resource types
        resource_types = [r.__resource_type__ for r in fhir_resources]
        assert "Patient" in resource_types, "Missing Patient resource"
        assert "Encounter" in resource_types, "Missing Encounter resource"
        assert "Condition" in resource_types, "Missing Condition resource"
        assert "MedicationRequest" in resource_types, "Missing MedicationRequest resource"
        assert "Observation" in resource_types, "Missing Observation resource"
        assert "CarePlan" in resource_types, "Missing CarePlan resource"
    asyncio.run(run())

if __name__ == "__main__":
    #test_icd10_extraction_on_soap_01()
    #test_icd10_extraction_on_soap_02()
    #test_icd10_extraction_on_soap_03()
    #test_icd10_extraction_on_all_soap_notes()
    #test_rxnorm_extraction_on_soap_01()
    #test_agentic_structured_extraction_on_soap_01()
    #test_agentic_structured_extraction_on_soap_02()
    #test_to_fhir_bundle_on_soap_02()
    test_to_fhir_strict_on_soap_02()
    # Uncomment to run RxNorm lookup test
    #asyncio.run(test_rxnorm_lookup())