from sqlalchemy import Column, Integer, MetaData, String
from sqlalchemy.orm import DeclarativeBase

metadata = MetaData(schema="knowledge")


class Base(DeclarativeBase):
    metadata = metadata


class Fact(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True)
    info = Column(String, nullable=False)
