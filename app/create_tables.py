from app.database import engine
from app.models import Base
from app import crud
from app.schemas import DocumentCreate
from sqlalchemy.orm import Session
from app.database import SessionLocal
import os

def create_tables():
    Base.metadata.create_all(bind=engine)
    
    # Add sample documents
    db = SessionLocal()
    try:
        # Check if documents already exist
        if not crud.get_documents(db):
            # First, add SOAP notes as they contain the most relevant patient information
            soap_notes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'soap_notes')
            for filename in os.listdir(soap_notes_dir):
                if filename.endswith('.txt'):
                    with open(os.path.join(soap_notes_dir, filename), 'r') as f:
                        content = f.read()
                        # Extract date from content if possible
                        date = "Unknown"
                        if "Encounter Date:" in content:
                            date = content.split("Encounter Date:")[1].split("\n")[0].strip()
                        doc = DocumentCreate(
                            title=f"SOAP Note - {filename} ({date})",
                            content=content
                        )
                        crud.create_document(db, document=doc)
            
            # Then add medical guidelines as reference material
            medical_guidelines = [
                DocumentCreate(
                    title="Diabetes Management Guidelines",
                    content="""
                    Type 2 Diabetes Management Protocol:
                    
                    Initial Assessment:
                    - Fasting blood glucose
                    - HbA1c
                    - Lipid profile
                    - Renal function
                    
                    First-line Treatment:
                    - Metformin 500mg twice daily
                    - Lifestyle modifications
                    - Dietary counseling
                    
                    Second-line Options:
                    - Sulfonylureas
                    - DPP-4 inhibitors
                    - GLP-1 receptor agonists
                    
                    Monitoring:
                    - HbA1c every 3 months
                    - Annual eye exam
                    - Annual foot exam
                    - Regular blood pressure checks
                    """
                ),
                DocumentCreate(
                    title="Hypertension Treatment Protocol",
                    content="""
                    Hypertension Management Guidelines:
                    
                    Classification:
                    - Normal: <120/<80
                    - Elevated: 120-129/<80
                    - Stage 1: 130-139/80-89
                    - Stage 2: ≥140/≥90
                    
                    First-line Medications:
                    - ACE inhibitors
                    - ARBs
                    - Calcium channel blockers
                    - Thiazide diuretics
                    
                    Contraindications:
                    - ACE inhibitors: Pregnancy, bilateral renal artery stenosis
                    - ARBs: Pregnancy
                    - Beta-blockers: Severe asthma, heart block
                    
                    Monitoring:
                    - Blood pressure every 3-6 months
                    - Renal function
                    - Electrolytes
                    """
                ),
                DocumentCreate(
                    title="Cardiac Care Guidelines",
                    content="""
                    Post-Cardiac Procedure Care:
                    
                    Immediate Post-Procedure:
                    - Monitor vital signs
                    - Check access site
                    - Assess for complications
                    
                    Medications:
                    - Antiplatelet therapy
                    - Statins
                    - Beta-blockers if indicated
                    
                    Follow-up Schedule:
                    - 1 week post-procedure
                    - 1 month
                    - 3 months
                    - 6 months
                    - Annual thereafter
                    
                    Lifestyle Recommendations:
                    - Cardiac rehabilitation
                    - Smoking cessation
                    - Dietary modifications
                    - Regular exercise
                    """
                )
            ]
            
            # Add guidelines to database
            for doc in medical_guidelines:
                crud.create_document(db, document=doc)
    finally:
        db.close() 