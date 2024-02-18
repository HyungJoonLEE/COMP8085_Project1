import pandas as pd
from scripts import objectCSV as ref

ORIGINAL_CSV = '../data/UNSW-NB15-BALANCED-TRAIN.csv'


def main():
    # Read csv using pandas in Latin mode
    df = pd.read_csv(ORIGINAL_CSV, encoding='ISO-8859-1', low_memory=False)
    df2 = ref.refactor_data(df)
    print(df2.dtypes)

    return 0

if __name__ == "__main__":
    main()
