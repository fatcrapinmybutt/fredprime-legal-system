from sqlalchemy import Column, Integer, MetaData, String
from sqlalchemy.orm import DeclarativeBase

metadata = MetaData(schema="forms")


class Base(DeclarativeBase):
    metadata = metadata


class Form(Base):
    __tablename__ = "forms"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
