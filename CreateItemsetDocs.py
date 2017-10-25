import data._Constants as const
import data.Pipelines as pip


def main():
    db = const.db
    itemsets = list(db.matchDetails.aggregate(pip.itemsets_adc(), allowDiskUse=True))
    print("Number of Itemsets to insert: " + str(len(itemsets)))
    db.itemsets_adc.insert_many(itemsets)
    print("Insert complete!")

if __name__ == "__main__":
    main()
