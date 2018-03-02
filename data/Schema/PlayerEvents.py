from sqlalchemy import *
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.declarative import declarative_base

global PlayerEvent
Base = declarative_base()

class ChampionStats(Base, object):
    __tablename__ = 'ChampionStats'