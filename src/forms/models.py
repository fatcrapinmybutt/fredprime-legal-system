from sqlalchemy import Column, Integer, MetaData, String
from sqlalchemy.orm import declarative_base

metadata = MetaData(schema="forms")
Base = declarative_base(metadata=metadata)


class Form(Base):
    __tablename__ = "forms"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
