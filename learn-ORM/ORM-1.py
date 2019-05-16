import sqlalchemy
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date, String, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

from numpy import genfromtxt
from time import time

Base = declarative_base()


class cdb1(Base):
    # Tell SQLAlchemy what the table name is and if there's any
    # table-specific arguments it should know about
    __tablename__ = 'cdb1'
    __table_args__ = {'sqlite_autoincrement': True}
    # tell SQLAlchemy the name of column and its attributes:
    id = Column(Integer, primary_key=True, nullable=False)
    Name = Column(VARCHAR(40))
    Shack = Column(VARCHAR)
    DB = Column(Integer)
    Payments = Column(Integer)
    Status = Column(VARCHAR)


# interface to db with logging
engine = create_engine('sqlite:///cdb.db', echo=True)
engine.connect()  # establish DB API connection to db
engine.execute()

Base.metadata.create_all(engine)
file_name = 'C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\learn-ORM\\client_db.csv'
df = pd.read_csv(file_name)
df.to_sql(con=engine, index_label='id', name=cdb1.__tablename__,
          if_exists='replace')

cdb1.__

declarative_base?
create_engine?
df.to_sql?
