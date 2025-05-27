import asyncio
import pytest
from app.services.llm.llm_service import LLMService
from fastapi import HTTPException
import json

# Initialize LLMService
llm_service = LLMService()

# Test data
VALID_NOTE = """
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

async def test_normal_summarization():
    """Test normal summarization of a valid medical note"""
    result = await llm_service.process_note(VALID_NOTE, tasks=["summarize"])
    summary = result["summarize"]
    assert isinstance(summary, str)
    assert len(summary) > 0
    print("\nNormal summarization test passed!")
    print(f"Summary: {summary[:200]}...")

async def test_empty_note():
    """Test handling of empty note"""
    with pytest.raises(HTTPException) as exc_info:
        await llm_service.process_note("", tasks=["summarize"])
    assert exc_info.value.status_code == 400
    print("\nEmpty note test passed!")

async def test_invalid_input():
    """Test handling of invalid input type"""
    with pytest.raises(HTTPException) as exc_info:
        await llm_service.process_note(123, tasks=["summarize"])  # Passing a number instead of string
    assert exc_info.value.status_code == 400
    print("\nInvalid input test passed!")

async def test_caching_behavior():
    """Test that identical notes return cached results"""
    # First call
    start_time = asyncio.get_event_loop().time()
    result1 = await llm_service.process_note(VALID_NOTE, tasks=["summarize"])
    first_call_time = asyncio.get_event_loop().time() - start_time

    # Second call with same note
    start_time = asyncio.get_event_loop().time()
    result2 = await llm_service.process_note(VALID_NOTE, tasks=["summarize"])
    second_call_time = asyncio.get_event_loop().time() - start_time

    # Verify results
    assert result1["summarize"] == result2["summarize"]
    assert second_call_time < first_call_time
    print("\nCaching behavior test passed!")
    print(f"First call time: {first_call_time:.2f} seconds")
    print(f"Second call time: {second_call_time:.2f} seconds")

async def test_different_notes():
    """Test that different notes get different summaries"""
    # Create a slightly different note
    different_note = VALID_NOTE + "\nAdditional Note: Patient reports mild chest discomfort."
    
    result1 = await llm_service.process_note(VALID_NOTE, tasks=["summarize"])
    result2 = await llm_service.process_note(different_note, tasks=["summarize"])
    
    assert result1["summarize"] != result2["summarize"]
    print("\nDifferent notes test passed!")
    print(f"Original summary: {result1['summarize'][:100]}...")
    print(f"Different summary: {result2['summarize'][:100]}...")

async def run_all_tests():
    """Run all test cases"""
    print("Starting summarize_note function tests...")
    
    try:
        await test_normal_summarization()
        await test_empty_note()
        await test_invalid_input()
        await test_caching_behavior()
        await test_different_notes()
        print("\nAll tests completed successfully! ✅")
    except Exception as e:
        print(f"\nTest failed: {str(e)} ❌")
        raise

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 