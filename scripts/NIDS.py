import pandas as pd
from scripts import preprocess as ref

ORIGINAL_CSV = '../data/UNSW-NB15-BALANCED-TRAIN.csv'


def main():
    # Read csv using pandas in Latin mode
    origin = pd.read_csv(ORIGINAL_CSV, encoding='ISO-8859-1', low_memory=False)
    df = ref.preprocess_data(origin)

    print(df.info())

    return 0

if __name__ == "__main__":
    main()
