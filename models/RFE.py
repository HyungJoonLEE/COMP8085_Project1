import warnings
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def rfe_model(training_data, test_data, validate_data, target):
    warnings.filterwarnings(action='ignore')

    x = training_data.drop(['attack_cat', 'Label'], axis=1)
    y = training_data[target]

    if target == 'Label':
        classifier = LogisticRegression()
    else:
        # attack_cat LogisticRegression() take so long => Decision Tree
        classifier = DecisionTreeClassifier()

    rfe = RFE(classifier, n_features_to_select=20, step=1)
    rfe.fit(x, y)

    # Print out selected features
    print("\nSelected feature names:", rfe.get_feature_names_out())

    # Rank the features (1 is the priority)
    print("Feature ranking:", rfe.ranking_)

    selected_features = np.append(rfe.get_feature_names_out(),
                                  ['attack_cat', 'Label'])

    train_df = training_data[selected_features]
    train_df.to_csv('data/RFE/label/RFE-train-bin.csv', index=False)

    test_df = test_data[selected_features]
    test_df.to_csv('data/RFE/label/RFE-test-bin.csv', index=False)
