import pandas as pd
from models import RFE as rfe
from sklearn import metrics # is used to create classification results
from sklearn.ensemble import GradientBoostingClassifier


def gbc_model(train_data, test_data, val_data, target):
    vulnerabilities = ["None", "Generic", "Fuzzers", "Exploits", "Dos",
                       "Reconnaissance", "Analysis", "Shellcode",
                       "Backdoors", "Worms"]
    # Call feature selected .csv using RFE
    ft_train_data, ft_test_data, ft_val_data = rfe.rfe_model(train_data,
                                                             test_data,
                                                             val_data,
                                                             target)

    ip_feature = ['srcip', 'dstip']
    for ip in ip_feature:
        train_data[ip] = pd.factorize(train_data[ip])[0]
        test_data[ip] = pd.factorize(test_data[ip])[0]
        val_data[ip] = pd.factorize(val_data[ip])[0]

    print("\nProcessing all feature - " + target)
    x_all_train = train_data.drop(['attack_cat', 'Label'], axis=1)
    y_all_train = train_data[target]
    x_all_test = test_data.drop(['attack_cat', 'Label'], axis=1)
    y_all_test = test_data[target]

    if target == 'Label':
        # Default: max_depth=3, learning_rate=0.1
        classifier_all = GradientBoostingClassifier(n_estimators=20,
                                                    learning_rate=0.5,
                                                    max_depth=3)
    else:
        classifier_all = GradientBoostingClassifier(learning_rate=0.1,
                                                    max_depth=3)
    classifier_all.fit(x_all_train, y_all_train)
    pred_all = classifier_all.predict(x_all_test)
    if target == 'Label':
        print(metrics.classification_report(y_all_test, pred_all))
    else:
        print(metrics.classification_report(y_all_test,
                                            pred_all,
                                            target_names=vulnerabilities))

    # Selected Feature
    print("\nProcessing selected feature - " + target)
    x_ft_train = ft_train_data.drop(['attack_cat', 'Label'], axis=1)
    y_ft_train = ft_train_data[target]
    x_ft_test = ft_test_data.drop(['attack_cat', 'Label'], axis=1)
    y_ft_test = ft_test_data[target]

    if target == 'Label':
        # Default: max_depth=3, learning_rate=0.1
        classifier_ft = GradientBoostingClassifier(n_estimators=20,
                                                   learning_rate=0.5,
                                                   max_depth=3)
    else:
        classifier_ft = GradientBoostingClassifier(learning_rate=0.1,
                                                   max_depth=3)

    classifier_ft.fit(x_ft_train, y_ft_train)
    pred_ft = classifier_ft.predict(x_ft_test)

    if target == 'Label':
        print(metrics.classification_report(y_ft_test, pred_ft))
    else:
        print(metrics.classification_report(y_ft_test,
                                            pred_ft,
                                            target_names=vulnerabilities))



