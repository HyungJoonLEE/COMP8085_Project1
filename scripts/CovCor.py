import pandas as pd
from scripts import preprocess as ref
import matplotlib.pyplot as plt

ORIGINAL_CSV = '../data/UNSW-NB15-BALANCED-TRAIN.csv'




def main():
    # Read csv using pandas in Latin mode
    df = pd.read_csv(ORIGINAL_CSV, encoding='ISO-8859-1', low_memory=False)
    df2 = ref.refactor_data(df)

    X = df2.sport.values
    Y = df2.Label.values

    print(X)
    plt.scatter(X, Y, alpha=0.5)
    plt.title('TARGET ~ BMI')
    plt.xlabel('sport')
    plt.ylabel('Label')
    # plt.show()

    return 0

if __name__ == "__main__":
    main()



