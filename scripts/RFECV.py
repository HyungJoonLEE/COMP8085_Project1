import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split


def rfe_model(df, drop):
    x = df.drop(['attack_cat', 'Label'], axis=1)
    y = df[drop]

    # Train model with 30% of data will be used as a test model
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=42)

    warnings.filterwarnings(action='ignore')

    estimator = LogisticRegression()

    """
    This is code for RFE
    # Number of feature to select = 6
    # Number of feature to delete every step = 1
    selector_Label = RFE(estimator_Label, n_features_to_select=6, step=1)
    """

    # Number of feature to delete every step = 1
    # Number of cross fold validation
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(x_train, y_train)

    selected_columns = x_train.columns[selector.support_]

    selected_columns
