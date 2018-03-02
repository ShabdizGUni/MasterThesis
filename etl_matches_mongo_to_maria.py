from data.constants import *
from data.Schema.PlayerStats import *
from sqlalchemy.orm import *
from datetime import datetime
from data.Schema.PlayerStats import PlayerDetail
import data.Pipelines as pl
import threading

start = datetime.now()


class DBWorker(threading.Thread):
    def __init__(self, platform):
        threading.Thread.__init__(self)
        self.platform = platform
        engine = create_engine('mysql+pymysql://root:Milch4321@localhost:3306/leaguestats', encoding='latin1',
                               echo=False)
        session_factory = sessionmaker(bind=engine)
        Session = scoped_session(session_factory)
        self.session = Session()
        md: MetaData = Base.metadata
        md.create_all(engine, checkfirst=True)

    def run(self):
        print(str(datetime.now()) + ": Worker " + self.platform + " started")
        for patch in PATCHES:
            startQuery = datetime.now()
            matchDocs = list(db.matchDetails.aggregate(pl.playerstats(self.platform, patch), allowDiskUse=True))
            print(str(datetime.now()) + " " + patch + " " + self.platform + ": Collection acquired in " + str(
                datetime.now() - startQuery) + "! Begin to insert into MariaDB")
            i = 1  # rows
            startInsert = datetime.now()
            for m in matchDocs:
                self.session.add(PlayerDetail(m))
                i += 1
                if (i % 1000) == 0: self.session.flush()  # flush every 1000 rows
            self.session.commit()
            matchDocs = None
            print(str(datetime.now()) + ": inserted " + self.platform + " " + patch + " in " + str(
                datetime.now()-startInsert))
        print(str(datetime.now()) + " Worker " + self.platform + " finished after " + str(datetime.now() - start))


def main():
    print(str(datetime.now()) + " started")
    workers = [DBWorker(platform) for platform in PLATFORMS]
    for w in workers: w.start()
    for w in workers: w.join()
    print("Finished in " + str(datetime.now() - start))


if __name__ == "__main__":
    main()
