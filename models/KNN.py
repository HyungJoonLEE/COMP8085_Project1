import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
from models import BKS

#from COMP8085_Project1.models.BKS import bks_model

LIMIT = 20000


def knn_model(training_data, test_data, validation_data, target):
    Y_train = training_data[target]
    X_train = training_data.drop(['srcip', 'dstip','Label', 'attack_cat'], axis=1)

    Y_test = test_data[target]
    X_test = test_data.drop(['srcip', 'dstip','Label', 'attack_cat'], axis=1)

    Y_validate = validation_data[target]
    X_validate = validation_data.drop(['srcip', 'dstip','Label', 'attack_cat'], axis=1)

    # 96% -> 99%
    # ~40% -> 46%
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_validate = scaler.transform(X_validate)

    record_val_accuracy = []
    # Find the best hyperparameter k based on validation set accuracy
    best_k, best_score = 0, 0

    if target == "Label":
        # Retrain the model with the best hyperparameter on the combined training and validation set
        final_model = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan', n_jobs=5)
        final_model.fit(np.concatenate([X_train, X_validate]), np.concatenate([Y_train, Y_validate]))
        Y_test_pred = final_model.predict(X_test)
        print("\n==========Label Scores for all features ==========")
        print(f"{classification_report(Y_test_pred,Y_test, labels=[0,1], zero_division=0)}")
    elif target == "attack_cat":
        # Retrain the model with the best hyperparameter on the combined training and validation set
        final_model = KNeighborsClassifier(n_neighbors=9, weights='distance', metric='manhattan', n_jobs=5)
        final_model.fit(np.concatenate([X_train, X_validate]), np.concatenate([Y_train, Y_validate]))
        Y_test_pred = final_model.predict(X_test)
        print("\n==========Attack Cat Scores for all features ==========")
        vulnerabilities = ["None", "Generic", "Fuzzers", "Exploits", "Dos", "Reconnaissance","Analysis","Shellcode","Backdoors","Worms"]
        print(f"{classification_report(Y_test_pred,Y_test,target_names=vulnerabilities, zero_division=0)}")

    bks_trainning_data, bks_test_data, bks_validation_data = bks_model(training_data, test_data, validation_data, target)
    Y_bks_train = bks_trainning_data[target]
    X_bks_train = bks_trainning_data.drop(['Label', 'attack_cat'], axis=1)
    Y_bks_test = bks_test_data[target]
    X_bks_test = bks_test_data.drop(['Label', 'attack_cat'], axis=1)
    Y_bks_val = bks_validation_data[target]
    X_bks_val = bks_validation_data.drop(['Label', 'attack_cat'], axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_bks_train)
    X_test = scaler.transform(X_bks_test)
    X_validate = scaler.transform(X_bks_val)

    if target == "Label":
        # Retrain the model with the best hyperparameter on the combined training and validation set
        final_model = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan', n_jobs=3)
        final_model.fit(np.concatenate([X_train, X_validate]), np.concatenate([Y_bks_train, Y_bks_val]))
        Y_test_pred = final_model.predict(X_test)
        print("\n==========Label Scores for SELECTED features ==========")
        print(f"{classification_report(Y_test_pred,Y_test, labels=[0,1], zero_division=0)}")
    elif target == "attack_cat":
        # Retrain the model with the best hyperparameter on the combined training and validation set
        final_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan', n_jobs=5)
        final_model.fit(np.concatenate([X_train, X_validate]), np.concatenate([Y_bks_train, Y_bks_val]))
        Y_test_pred = final_model.predict(X_test)
        print("\n==========Attack Cat Scores for SELECTED features ==========")
        vulnerabilities = ["None", "Generic", "Fuzzers", "Exploits", "Dos", "Reconnaissance","Analysis","Shellcode","Backdoors","Worms"]
        print(f"{classification_report(Y_test_pred,Y_test,target_names=vulnerabilities, zero_division=0)}")


