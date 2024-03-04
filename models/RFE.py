import warnings
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

    model = DecisionTreeClassifier()
    rfe = RFE(model, n_features_to_select=20, step=1)
    rfe.fit(x, y)

    print("\nSelected feature names:", rfe.get_feature_names_out())
    print("Feature ranking:", rfe.ranking_)

