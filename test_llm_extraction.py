import asyncio
from app.services.llm.llm_service import LLMService

# 3.5 is not able to extract the conditions correctly, so we are using GPT-4 for now
async def test_llm_extraction_on_soap_02():
    with open("soap_notes/soap_02.txt", "r") as f:
        note_text = f.read()
    llm_service = LLMService()
    entities = await llm_service.extract_entities(note_text)
    print("Extracted conditions:", entities.get("conditions"))
    print("Full LLM output:")
    import json
    print(json.dumps(entities, indent=2))

# GPT-4 is able to extract the conditions correctly, so we are using it for now
def test_llm_extraction_on_soap_02_gpt4():
    async def run():
        with open("soap_notes/soap_02.txt", "r") as f:
            note_text = f.read()
        llm_service = LLMService()
        entities = await llm_service.extract_entities(note_text, model="gpt-4")
        print("[GPT-4] Extracted conditions:", entities.get("conditions"))
        print("[GPT-4] Full LLM output:")
        import json
        print(json.dumps(entities, indent=2))
    asyncio.run(run())

if __name__ == "__main__":
    asyncio.run(test_llm_extraction_on_soap_02())
    # Uncomment to run GPT-4 test
    #test_llm_extraction_on_soap_02_gpt4() 