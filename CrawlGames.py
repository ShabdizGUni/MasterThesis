from Lib.Crawler import Crawler

def main():
    matches = 10000

    naCrawler = Crawler(region="na1", matchesPerPatch=matches)
    brCrawler = Crawler(region="br1", matchesPerPatch=matches)
    krCrawler = Crawler(region="kr", matchesPerPatch=matches)
    euwCrawler = Crawler(region="euw1", matchesPerPatch=matches)

    crawler = [naCrawler, brCrawler, krCrawler, euwCrawler]

    for c in crawler: c.start()

    for c in crawler: c.join()

if __name__ == "__main__":
    main()
