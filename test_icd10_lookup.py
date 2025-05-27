import asyncio
import glob
import os
import logging
from datetime import datetime
from app.services.llm.llm_service import LLMService
from app.services.extraction.code_lookup.icd_service import ICDLookupService

# Set up logging to file
logging.basicConfig(
    filename='icd10_lookup_test.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

async def test_icd10_lookup_on_all_soap_notes():
    note_files = sorted(glob.glob("soap_notes/soap_*.txt"))
    llm_service = LLMService()
    icd_service = ICDLookupService()
    for file in note_files:
        print(f"\n==== {file} ====")
        with open(file, "r") as f:
            note_text = f.read()
        entities = await llm_service.extract_entities(note_text)
        conditions = entities.get("conditions", [])
        print(f"Conditions extracted by LLM: {conditions}")
        for cond in conditions:
            result = await icd_service.agentic_icd10_lookup(cond)
            print(f"Condition: {cond}")
            print(f"  ICD-10 Code: {result.get('icd10_code')}")
            print(f"  Description: {result.get('description')}")
            print(f"  Confidence: {result.get('confidence')}")
            print(f"  Agent Reasoning: {result.get('agent_reasoning')}")
            print()

def test_icd10_lookup_on_conditions():
    async def run():
        icd_service = ICDLookupService()
        test_conditions = [
            "Low HDL",
            "Meniscal tear",
            "Hypertension",
            "Type 2 diabetes",
            "Hyperlipidemia",
            "Chronic kidney disease",
            "Visual discomfort",
            "Arthroscopic meniscal repair"
        ]
        for cond in test_conditions:
            start_time = datetime.now()
            result = await icd_service.agentic_icd10_lookup(cond)
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            log_lines = [
                f"\n--- ICD-10 Lookup for: {cond} ---",
                f"Timestamp: {start_time.isoformat()}",
                f"Elapsed: {elapsed:.2f} seconds",
                f"ICD-10 Code: {result.get('icd10_code')}",
                f"Description: {result.get('description')}",
                f"Confidence: {result.get('confidence')}",
                f"Agent Reasoning: {result.get('agent_reasoning', '')}",
                f"Validation Flags: {result.get('validation_flags', [])}"
            ]
            log_output = "\n".join(log_lines)
            print(log_output)
            logging.info(log_output)
    asyncio.run(run())

def test_icd10_lookup_on_soap_02():
    async def run():
        llm_service = LLMService()
        icd_service = ICDLookupService()
        note_path = os.path.join("soap_notes", "soap_02.txt")
        with open(note_path, "r") as f:
            note_text = f.read()
        entities = await llm_service.extract_entities(note_text, model="gpt-4")
        conditions = entities.get("conditions", [])
        print(f"\n==== soap_02.txt ====")
        print(f"Conditions extracted by LLM: {conditions}")
        for cond in conditions:
            start_time = datetime.now()
            result = await icd_service.agentic_icd10_lookup(cond)
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            log_lines = [
                f"\n--- ICD-10 Lookup for: {cond} ---",
                f"Timestamp: {start_time.isoformat()}",
                f"Elapsed: {elapsed:.2f} seconds",
                f"ICD-10 Code: {result.get('icd10_code')}",
                f"Description: {result.get('description')}",
                f"Confidence: {result.get('confidence')}",
                f"Agent Reasoning: {result.get('agent_reasoning', '')}",
                f"Validation Flags: {result.get('validation_flags', [])}"
            ]
            # Log top suggestions if low confidence or no code
            if (not result.get('icd10_code') or result.get('confidence', 0) < 0.5) and 'top_suggestions' in result:
                log_lines.append("Top ICD-10 Suggestions:")
                for s in result['top_suggestions']:
                    log_lines.append(f"  Code: {s['code']}, Desc: {s['description']}, Similarity: {s['similarity_score']:.3f}")
            log_output = "\n".join(log_lines)
            print(log_output)
            logging.info(log_output)
    asyncio.run(run())

if __name__ == "__main__":
    #asyncio.run(test_icd10_lookup_on_all_soap_notes())
    #test_icd10_lookup_on_conditions()
    test_icd10_lookup_on_soap_02() 