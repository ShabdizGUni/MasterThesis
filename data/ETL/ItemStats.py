from pymongo import *
from data.constants import *
from pprint import pprint
from sqlalchemy import *
from sqlalchemy.orm import *
from data.Schema.ItemStats import *

def main():
    engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1', echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    md: MetaData = Base.metadata
    md.create_all(engine)
    itemDicts = db.items.find({})
    for itemDict in itemDicts:
        data = itemDict["data"]
        version = itemDict["version"]
        items = [v for k, v in data.items()]
        for i in items:
            if i["maps"]["11"]:
                i["version"] = version
                session.add(ItemStats(i))
        session.commit()

if __name__ == "__main__":
    main()