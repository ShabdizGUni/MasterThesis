from sqlalchemy.orm import *

from Lib import Schema
from Lib.Schema import *

engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/sqlalchemytest', encoding='latin1', echo=True)
Session = sessionmaker(bind=engine)
md: MetaData = Schema.Base.metadata
md.create_all(engine)


