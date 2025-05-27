# Medical Documents API

A FastAPI-based application for managing and processing medical documents with AI-powered analysis, agentic extraction, and FHIR-compatible output. This project implements a medical document processing system in multiple steps, closely following the assignment requirements.

---

## Assignment Mapping & Progress

This project is structured to address each part of the assignment:

### **Part 1: FastAPI Backend Foundation**
- FastAPI application with health check endpoint (`GET /health`)
- SQLite database with SQLAlchemy ORM
- Document model with CRUD operations
- RESTful API endpoints for document management (`/documents`)
- Error handling and input validation

### **Part 2: LLM API Integration**
- OpenAI API integration (model-agnostic, can be configured)
- Medical note processing endpoint (`POST /process_note`)
- Multiple processing tasks: summarize, paraphrase, extract complaint
- Environment variable configuration for API keys
- Response caching with Redis
- **Model selection via config file:** You can specify which LLM model/provider to use in the `config.json` file or via environment variables. This enables easy switching between different LLMs for experimentation or cost optimization.

### **Part 3: Retrieval-Augmented Generation (RAG) Pipeline**
- Document embedding and storage with ChromaDB
- Semantic search implementation
- Context retrieval system
- Document chunking and cleaning
- Source citation and relevance scoring
- Question answering with medical context (`POST /answer_question`)

### **Part 4: Agent for Structured Data Extraction**
- Extraction service for patient info, conditions, medications, vitals, labs, and plan
- ICD-10 and RxNorm lookups (direct, semantic, LLM fallback)
- Pydantic model validation
- Detailed logging
- Endpoint: `POST /extract_structured`

#### **Detailed Extraction & Coding Pipeline**

1. **Entity Extraction**
   - The agent receives a raw medical note (typically in SOAP format).
   - It parses the note to extract structured entities:
     - Patient information (name, age, gender, etc.)
     - Chief complaint
     - Diagnoses/conditions (from Assessment section)
     - Medications (from Medications/Plan section)
     - Vital signs and lab results (from Objective section)
     - Plan items (follow-up, tests, referrals, etc.)
   - Extraction is performed using a combination of regular expressions, rule-based parsing, and LLM prompts for ambiguous or complex cases.

2. **ICD-10 and RxNorm Code Lookup**
   - For each extracted condition, the system attempts to assign an ICD-10 code.
   - For each medication, it attempts to assign an RxNorm code.
   - The lookup process is multi-stage:
     - **Direct Extraction:** If a code is already present in the note (e.g., "Hypertension (ICD-10: I10)"), it is extracted and used directly.
     - **API Lookup:** If no code is present, the system queries public APIs (NIH Clinical Tables for ICD-10, RxNav for RxNorm) to find candidate codes and descriptions.

3. **Fuzzy and Contextual Similarity Matching**
   - If the API returns multiple candidates or confidence is low, the system computes semantic similarity between the extracted condition/medication and the candidate descriptions.
   - Embeddings (e.g., from OpenAI) is used to compare the text of the condition/medication with candidate codes.
   - The candidate with the highest similarity score above a threshold is selected.
   - Top suggestions and their scores are logged for transparency.

4. **LLM Fallback**
   - If no suitable code is found via direct extraction or API/embedding similarity, the system prompts the LLM to suggest the most likely code and description based on the extracted entity and context.
   - The LLM's suggestion is used as a last resort, and a flag is set in the output to indicate LLM fallback was used.

5. **Post-Processing and Output Construction**
   - All extracted and coded entities are validated and structured using Pydantic models.
   - Confidence scores, mapping suggestions, and fallback flags are included in the output for each entity.
   - The final output is a comprehensive JSON object containing all structured data, codes, and metadata, ready for downstream use or FHIR conversion.

6. **Logging and Transparency**
   - Each step of the extraction and coding process is logged with timestamps and confidence levels.
   - Suggestions and fallback usage are included in the output for review and auditability.

#### **About Confidence Scores and Coding Strategy**

**Confidence Score:**
- Each extracted code (ICD-10 or RxNorm) is assigned a confidence score.
- This score reflects how certain the system is that the code matches the extracted entity.
- The confidence score is determined by:
  - Whether the code was found directly in the note (highest confidence)
  - The match quality from API lookups (e.g., if the API returns a single, exact match)
  - The similarity score from embedding-based (semantic) matching (normalized to [0,1])
  - If the LLM is used as a fallback, the confidence is typically lower and flagged as such
- Including a confidence score helps:
  - Indicate to downstream users or systems how reliable the code assignment is
  - Support transparency and explainability in the extraction process
  - Allow for thresholding or manual review of low-confidence mappings

**Why Not Just Use LLM for ICD-10 Coding?**
- LLMs are powerful for language understanding, but:
  - They may hallucinate codes or return outdated/incomplete information
  - They are not guaranteed to be up-to-date with the latest medical coding standards
  - They lack the authoritative, curated mapping that official APIs and code sets provide
- The multi-stage approach (direct extraction, API lookup, semantic similarity, LLM fallback) ensures:
  - Maximum accuracy and compliance with real-world coding standards
  - Use of LLM only as a last resort, when structured and curated sources fail
  - More robust, explainable, and auditable results

### **Part 5: FHIR-Compatible Output**
- FHIR resource mapping (Patient, Condition, MedicationRequest, Observation, CarePlan)
- Data transformation from structured extraction to FHIR-like JSON
- Endpoint: `POST /to_fhir`
- (Stretch) Optionally uses `fhir.resources` for stricter compliance

### **Part 6: Containerization & Docker Compose**
- Dockerfile for FastAPI app
- docker-compose.yml for orchestrating app, Redis, and ChromaDB
- .env file for secrets
- Hot-reload support for development
- Persistent DB volume

---

## Features

- **CRUD Operations** for medical documents
- **AI Processing**: Summarization, paraphrasing, complaint extraction
- **RAG Pipeline**: Contextual Q&A with source citation
- **Agentic Extraction**: Structured data extraction with code lookups
- **FHIR Output**: Converts structured data to FHIR-like resources
- **Caching**: Redis for LLM response caching
  - Redis is used to cache the results of expensive or repeated LLM and code lookup API calls. When a request (e.g., summarization, extraction, or code lookup) is made, the system first checks Redis to see if a cached response exists for the same input. If so, it returns the cached result instantly, reducing latency, API costs, and the risk of hitting rate limits. If not, it performs the operation and stores the result in Redis for future use. This is especially valuable for repeated queries or when processing large batches of similar notes.
  - Redis is preferred because it is extremely fast (in-memory), supports flexible expiration policies (TTL), and is widely used in production for caching and session management. It helps ensure the system remains responsive and cost-effective, even under heavy load.
- **Model-Agnostic LLM Integration**: Configurable model selection
  - The project supports selecting different LLM providers or models via a configuration file (e.g., `config.json` or environment variables). This allows you to switch between OpenAI, Azure, or other providers without changing the codebase. You can specify the model name, API endpoint, and credentials in the config file. This design makes the system flexible, future-proof, and easy to adapt to new models or providers as they become available.
- **Containerized**: Easy local deployment with Docker Compose

---

## Project Structure

```
Project/
├── app/
│   ├── config.py
│   ├── crud.py
│   ├── create_tables.py
│   ├── database.py
│   ├── documents.db
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── services/
│       ├── cache_service.py
│       ├── chroma_db/
│       ├── extraction/
│       │   ├── extraction_service.py
│       │   ├── extraction_schemas.py
│       │   └── code_lookup/
│       │       ├── icd_service.py
│       │       ├── rxnorm_service.py
│       │   └── extraction_service.py
│       ├── llm/
│       │   └── llm_service.py
│       ├── rag/
│       │   ├── rag_service.py
│       │   ├── document/
│       │   │   └── document_service.py
│       │   ├── vector_db/
│       │   │   └── chroma_service.py
│       │   └── embedding/
│       │       └── embedding_service.py
│       └── redis_service.py
├── soap_notes/
│   ├── soap_01.txt
│   ├── soap_02.txt
│   ├── soap_03.txt
│   ├── soap_04.txt
│   ├── soap_05.txt
│   └── soap_06.txt
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── test_extraction.py
├── test_icd10_lookup.py
├── test_llm_extraction.py
├── test_rag.py
├── test_summarize.py
```

---

## Architecture Diagram

```mermaid
graph TD
    A[User / API Client]
    A -->|HTTP Requests| B(FastAPI Application)
    B -->|CRUD| C[(Relational DB<br>(SQLite/PostgreSQL))]
    B -->|Cache| D[(Redis)]
    B -->|RAG Pipeline| E[Vector DB<br>(ChromaDB/Pinecone)]
    B -->|LLM Calls| F[LLM API<br>(OpenAI, etc.)]
    B -->|Extraction Agent| G[Extraction Service]
    G -->|ICD-10/RxNorm Lookup| H[External Code APIs<br>(NIH, RxNav)]
    G -->|LLM Fallback| F
    B -->|FHIR Mapping| I[FHIR Output]
    I -->|Response| A

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#fff,stroke:#333,stroke-width:1px
    style D fill:#fff,stroke:#333,stroke-width:1px
    style E fill:#fff,stroke:#333,stroke-width:1px
    style F fill:#fff,stroke:#333,stroke-width:1px
    style G fill:#fff,stroke:#333,stroke-width:1px
    style H fill:#fff,stroke:#333,stroke-width:1px
    style I fill:#cfc,stroke:#333,stroke-width:1px
```

---

## Setup & Local Deployment

### **Docker Compose (Recommended)**

1. **Environment Variables**
   - Create a `.env` file in the Project directory:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
2. **Build and Run**
   ```bash
   cd Project
   docker-compose up --build
   ```
   - The API will be available at [http://localhost:8000](http://localhost:8000)

### **Local Development**

1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Initialize the database**
   ```bash
   python -m app.create_tables
   ```
4. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

---

## API Endpoints & Testing

### **Part 1: Document Management**
- `GET /health` - Health check
- `GET /documents/` - List all documents
- `GET /documents/{id}` - Get specific document
- `POST /documents/` - Create new document
- `PUT /documents/{id}` - Update document
- `DELETE /documents/{id}` - Delete document

**Test:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/documents/
```

### **Part 2: LLM API Integration**
- `POST /process_note` - Process medical note
  - Input: `{ "note_text": "...", "tasks": ["summarize", "paraphrase", "extract_complaint"] }`
  - Output: JSON with results for each task

**Test:**
```bash
curl -X POST http://localhost:8000/process_note -H "Content-Type: application/json" -d '{"note_text": "...", "tasks": ["summarize"]}'
```

### **Part 3: RAG Pipeline**
- `POST /answer_question` - Answer medical questions using RAG
  - Input: `{ "question": "..." }`
  - Output: `{ "answer": "...", "sources": [ ... ] }`

**Test:**
```bash
curl -X POST http://localhost:8000/answer_question -H "Content-Type: application/json" -d '{"question": "What is the patient's cholesterol history?"}'
```

### **Part 4: Agentic Extraction**
- `POST /extract_structured` - Extract structured data from a medical note
  - Input: `{ "note_text": "..." }`
  - Output: Structured JSON with patient info, conditions, medications, vitals, labs, plan, codes, etc.

**Test:**
```bash
curl -X POST http://localhost:8000/extract_structured -H "Content-Type: application/json" -d '{"note_text": "..."}'
```

### **Part 5: FHIR-Compatible Output**
- `POST /to_fhir` - Convert structured data to FHIR-like JSON
  - Input: Output from `/extract_structured`
  - Output: FHIR-style JSON bundle (Patient, Condition, Medication, Observation, CarePlan)

**Test:**
```bash
curl -X POST http://localhost:8000/to_fhir -H "Content-Type: application/json" -d '{...structured data...}'
```

---

## Sample Medical Notes (SOAP Format)

Sample notes are provided in `soap_notes/` directory. These can be used for testing extraction, RAG, and FHIR conversion. You may also create your own notes.

- `soap_01.txt` ... `soap_06.txt`

---

## FHIR Mapping Details

- **Patient**: Extracted from note (name, gender, age if available)
- **Condition**: Each diagnosis/condition with ICD-10 code
- **MedicationRequest**: Each medication with RxNorm code
- **Observation**: Vitals and lab results
- **CarePlan**: Plan items (follow-up, tests, etc.)

See `app/main.py` for mapping logic (`to_fhir_bundle`, `to_fhir_strict_resources`).

---

## Testing

### **Running Tests**
```bash
pytest test/                # Run all tests
pytest test/test_rag.py     # Run a specific test file
```

### **Testing Tips**
- Test each endpoint with provided SOAP notes
- Validate FHIR output structure
- Check error handling (404, 422, 500)

---

## Troubleshooting

- Ensure Redis and ChromaDB are running (see docker-compose)
- For FHIR mapping, check that structured extraction output is valid
- For LLM errors, verify API key and network

## Scalability Considerations

This project is designed with several scalable components and can be further improved for high-throughput or production environments:

### **Current Scalable Features**
- **FastAPI**: Supports asynchronous request handling, allowing concurrent processing of multiple API calls.
- **Containerization**: Docker and Docker Compose make it easy to deploy and scale services across multiple hosts or cloud environments.
- **Redis Caching**: Reduces repeated LLM/API calls, improving response time and lowering external API usage.
- **Modular Service Design**: Extraction, RAG, and LLM integration are separated, making it easier to scale or replace individual components.

### **How to Further Improve Scalability**
- **Horizontal Scaling**: Run multiple instances of the FastAPI app behind a load balancer (e.g., NGINX, AWS ALB) to handle more concurrent users.
- **Distributed Vector Database**: Use a scalable vector DB (e.g., managed Chroma, Pinecone, Weaviate) for RAG, supporting large document sets and high query volume.
- **Async Task Queues**: Offload heavy or long-running tasks (e.g., LLM calls, batch extraction) to background workers using Celery, RQ, or FastAPI's background tasks.
- **Database Optimization**: Move from SQLite to a production-grade DB (e.g., PostgreSQL) for better concurrency and reliability.
- **API Rate Limiting & Throttling**: Protect external APIs and your own service from overload by implementing rate limits (e.g., with FastAPI middleware or API gateway).
- **Autoscaling in Cloud**: Deploy on Kubernetes or a cloud platform with autoscaling to dynamically adjust resources based on load.
- **Monitoring & Observability**: Integrate logging, metrics, and tracing (e.g., Prometheus, Grafana, ELK stack) to monitor performance and identify bottlenecks.

By combining these strategies, the system can efficiently handle increased load, larger datasets, and more complex workflows in real-world healthcare or enterprise settings.

---

## License
MIT

---  