from data._Constants import *
from data.Schema.PlayerStats import *
from sqlalchemy.orm import *
from data.Pipelines import *
from datetime import datetime

from data.Schema.PlayerStats import PlayerDetail


def main():
    engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1', echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    md: MetaData = Base.metadata
    md.create_all(engine,checkfirst=True)
    for platform in PLATFORMS:
        #for patch in PATCHES:
        for patch in ["7.14"]:
            matchDocs = list(db.matchDetails2000.aggregate(Pipeline.proplayerstats(platform, patch), allowDiskUse=True))

            for match in matchDocs:
                session.add(PlayerDetail(match))

            session.commit()
            print(str(datetime.now())+ ": Inserted" + str(platform) + " " + str(patch))

if __name__ == "__main__":
    main()
