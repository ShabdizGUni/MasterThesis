from pymongo import *
from data._Constants import *
from pprint import pprint
from sqlalchemy import *
from sqlalchemy.orm import *
from data.Schema.ChampionStats import *

def main():
    engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1', echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    md: MetaData = Base.metadata
    md.create_all(engine)
    championDicts = db.champions.find({})
    for champDict in championDicts:
        data = champDict["data"]
        version = champDict["version"]
        champions = [v for k, v in data.items()]
        for c in champions:
            c["version"] = version
            session.add(ChampionStats(c))
            #db.championStats.insert_one(c)
        session.commit()

if __name__ == "__main__":
    main()

