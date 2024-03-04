from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def gbc_model(df, drop):
    x = df.drop(['attack_cat', 'Label'], axis=1)
    y = df[drop]

    # Train model with 30% of data will be used as a test model
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=y,
                                                        random_state=42)

    # Default: max_depth=3, learning_rate=0.1
    gbc = GradientBoostingClassifier(random_state=0,
                                     max_depth=3,
                                     learning_rate=0.1)
    gbc.fit(x_train, y_train)

    # Train set accuracy
    score_train = gbc.score(x_train, y_train)
    print('{:.3f}'.format(score_train))

    # Generalization accuracy
    score_test = gbc.score(x_test, y_test)
    print('{:.3f}'.format(score_test))

    # Visualize the result
    n_feature = x.shape[1]
    index = np.arange(n_feature)
    plt.barh(index, gbc.feature_importances_, align='center')
    plt.yticks(index, df.feature_names)
    plt.xlabel('feature importance', size=15)
    plt.ylabel('feature', size=15)
    plt.figure(figsize=(10, 10))
    plt.show()