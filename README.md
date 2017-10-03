# LeagueCrawlerProject

Project for my master thesis in order to fetch a set of games from four major regions Korea, EU West, Brazil and North America.
Core functionalities lie in CrawlGames.py and Lib.Crawler.py. In addition to that, FetchStaticData.py takes care of fetching data from the static data api endpoint.

The project at hand also inhibits some ETL-scripts for transforming documents from MongoDB to atomic records in preperation to get inserted into a Relational Database System (in this case MariaDB).

In addition to that, a R-Project is also embedded for data analysis and visualisation purposes.


# Requirements for fetching Match Details:
- MongoDB accessible at localhost:27017
  - Database called "LeagueCrawler", Collection called "matchDetails"
  
 # Requirements for transforming Documents into Records:
- MongoDB (see point above)
- MariaDB/MySQLDB accessible at localhost:3306

