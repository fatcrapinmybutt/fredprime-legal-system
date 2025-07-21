from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, MetaData

metadata = MetaData(schema="knowledge")
Base = declarative_base(metadata=metadata)

class Fact(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True)
    info = Column(String, nullable=False)
