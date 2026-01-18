from sqlalchemy import Column, Integer, MetaData, String
from sqlalchemy.orm import declarative_base

metadata = MetaData(schema="knowledge")
Base = declarative_base(metadata=metadata)


class Fact(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True)
    info = Column(String, nullable=False)
