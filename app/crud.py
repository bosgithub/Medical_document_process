from sqlalchemy.orm import Session
from app import models, schemas
from datetime import datetime

def get_document(db: Session, document_id: int):
    return db.query(models.Document).filter(models.Document.id == document_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Document).offset(skip).limit(limit).all()

def create_document(db: Session, document: schemas.DocumentCreate):
    current_time = datetime.utcnow()
    db_document = models.Document(
        **document.model_dump(),
        created_at=current_time,
        updated_at=current_time
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def update_document(db: Session, document_id: int, document: schemas.DocumentCreate):
    db_document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if db_document:
        for key, value in document.model_dump().items():
            setattr(db_document, key, value)
        db_document.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_document)
    return db_document

def delete_document(db: Session, document_id: int):
    db_document = db.query(models.Document).filter(models.Document.id == document_id).first()
    if db_document:
        db.delete(db_document)
        db.commit()
        return True
    return False 