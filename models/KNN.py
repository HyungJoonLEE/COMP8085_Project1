
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support


def knn_model(training_data=None, test_data=None, validation_data=None,target=None):
    # categories = ['srcip', 'dstip']
    # for category in categories:
    #     if category in training_data.columns:
    #         training_data.loc[:, [category]] = training_data.loc[:, [category]].apply(lambda x: pd.factorize(x)[0])
    #         test_data.loc[:, [category]] = test_data.loc[:, [category]].apply(lambda x: pd.factorize(x)[0])
    #         validation_data.loc[:, [category]] = validation_data.loc[:, [category]].apply(lambda x: pd.factorize(x)[0])

    Y_train = training_data[target]
    X_train = training_data.drop(['srcip', 'dstip','Label', 'attack_cat'], axis=1)

    Y_test = test_data[target]
    X_test = test_data.drop(['srcip', 'dstip','Label', 'attack_cat'], axis=1)

    Y_validate = validation_data[target]
    X_validate = validation_data.drop(['Label', 'attack_cat'], axis=1)

    record_val_accuracy = []

    # Find the best hyperparameter k based on validation set accuracy
    best_k, best_score = 0, 0
    for k in range(1, 5):
        neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
        neigh.fit(X_train, Y_train)
        score = neigh.score(X_train, Y_train)
        record_val_accuracy.append(score)
        if score > best_score:
            best_k, best_score = k, score

    if target == "Label":
        print(f"Best K based on validation accuracy: {best_k}")
        # Retrain the model with the best hyperparameter on the combined training and validation set
        final_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='manhattan')
        final_model.fit(pd.concat([X_train, X_validate]), pd.concat([Y_train, Y_validate]))
        Y_test_pred = final_model.predict(X_test)

        # Calculate and print performance metrics for the test set
        test_accuracy = final_model.score(X_test, Y_test)
        wei_precision, wei_recall, wei_f1_score, _ = precision_recall_fscore_support(Y_test, Y_test_pred,
                                                                                     average='weighted')
        mar_precision, mar_recall, mar_f1_score, _ = precision_recall_fscore_support(Y_test, Y_test_pred, average='macro')

        print("\n========== Scores after hyperparameter tuning ==========")
        print(f"Test Accuracy: {test_accuracy:.3f}")
        print(f"Macro Average Precision: {mar_precision:.3f}, Recall: {mar_recall:.3f}, F1-Score: {mar_f1_score:.3f}")
        print(f"Weighted Average Precision: {wei_precision:.3f}, Recall: {wei_recall:.3f}, F1-Score: {wei_f1_score:.3f}")

    elif target == "attack_cat":
        pass

