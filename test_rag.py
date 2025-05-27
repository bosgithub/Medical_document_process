import asyncio
import pytest
from app.services.rag.rag_service import RAGService
from fastapi import HTTPException
from app.services.cache_service import CacheService, cache
from app.services.redis_service import redis_client

# Initialize RAGService
rag_service = RAGService()

# Test data
TEST_DOCUMENTS = [
    {
        "id": 1,
        "title": "Patient Note 1",
        "content": """
        Patient: John Doe
        Date: 2024-03-20
        
        Chief Complaint: Patient presents with severe headache and fever for the past 3 days.
        
        History of Present Illness:
        - Started with mild headache 3 days ago
        - Fever developed on day 2
        - Temperature ranging from 100-102°F
        - No recent travel or exposure to sick contacts
        
        Physical Examination:
        - Temperature: 101.2°F
        - Blood Pressure: 120/80
        - Heart Rate: 88 bpm
        - Respiratory Rate: 16
        - General: Alert, oriented, in mild distress
        
        Assessment:
        1. Acute viral syndrome
        2. Possible influenza
        
        Plan:
        1. Rest and hydration
        2. Acetaminophen for fever and pain
        3. Follow up in 3 days if symptoms persist
        """
    },
    {
        "id": 2,
        "title": "Patient Note 2",
        "content": """
        Patient: Jane Smith
        Date: 2024-03-21
        
        Chief Complaint: Patient presents with persistent cough and fatigue for 2 weeks.
        
        History of Present Illness:
        - Dry cough started 2 weeks ago
        - Progressive fatigue
        - No fever
        - No recent travel
        
        Physical Examination:
        - Temperature: 98.6°F
        - Blood Pressure: 118/75
        - Heart Rate: 72 bpm
        - Respiratory Rate: 18
        - Lungs: Clear to auscultation
        
        Assessment:
        1. Post-viral cough
        2. Fatigue
        
        Plan:
        1. Rest and hydration
        2. Over-the-counter cough suppressant
        3. Follow up in 1 week if symptoms persist
        """
    }
]

async def test_basic_question_answering():
    """Test basic question answering functionality"""
    question = "What are the symptoms of John Doe?"
    result = await rag_service.answer_question(question, TEST_DOCUMENTS)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0
    
    print("\nBasic question answering test passed!")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

async def test_empty_question():
    """Test handling of empty question"""
    result = await rag_service.answer_question("", TEST_DOCUMENTS)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["answer"], str)
    assert result["answer"].strip() != ""
    print("\nEmpty question test passed!")
    print("Empty question answer:", result["answer"])

async def test_no_relevant_documents():
    """Test handling of question with no relevant documents"""
    question = "What is the capital of France?"
    result = await rag_service.answer_question(question, TEST_DOCUMENTS)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert result["answer"].strip() != ""
    print("\nNo relevant documents test passed!")
    print("No relevant documents answer:", result["answer"])
    print("No relevant documents sources:", result["sources"])

async def test_multiple_relevant_documents():
    """Test question that spans multiple documents"""
    question = "What are the common symptoms across all patients?"
    result = await rag_service.answer_question(question, TEST_DOCUMENTS)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 1
    
    print("\nMultiple relevant documents test passed!")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

async def test_specific_medical_details():
    """Test question about specific medical details"""
    question = "What was John Doe's temperature and blood pressure?"
    result = await rag_service.answer_question(question, TEST_DOCUMENTS)
    
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) >= 1
    assert any("John Doe" in s["content"] for s in result["sources"])
    
    print("\nSpecific medical details test passed!")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

async def run_all_tests():
    """Run all test cases"""
    print("Starting RAG service tests...")
    
    try:
        await test_basic_question_answering()
        await test_empty_question()
        await test_no_relevant_documents()
        await test_multiple_relevant_documents()
        await test_specific_medical_details()
        print("\nAll tests completed successfully! ✅")
    except Exception as e:
        print(f"\nTest failed: {str(e)} ❌")
        raise

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 