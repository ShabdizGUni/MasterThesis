from data.constants import *
from data.Schema.ProPlayerStats import *
from sqlalchemy.orm import *
from data.Pipelines import *
from datetime import datetime
from data.Schema.ProPlayerStats import ProPlayerDetail
from pathlib import Path

p = Path(__file__).parents[1]
# /absolute/path/to/two/levels/up

def main():
    engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1', echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    md: MetaData = Base.metadata
    md.create_all(engine, checkfirst=True)
    for patch in PATCHES:
        matchDocs = list(db.matchDetailsPro.aggregate(proplayerstats(patch), allowDiskUse=True))

        for match in matchDocs:
            session.add(ProPlayerDetail(match))

        session.commit()
        print(str(datetime.now()) + ": Inserted " + str(patch))


if __name__ == "__main__":
    main()
