from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from models.CCA import *

def SVM(training_data, test_data, validate_data, target):
    print("initializing targets")

    param_grid = {
        'C': [1, 10, 100, 1000],
        'kernel':['rbf']
    }

    targets = ["srcip","dstip","attack_cat", "Label"]
    #targets = ["attack_cat", "Label"]
    
    CCA_train, CCA_test, CCA_validate = correlation_coefficient(training_data, test_data, validate_data, target)
    
    sample1 = training_data.sample(n=20000)
    sample2 = test_data.sample(n=6000)

    #sample1 = CCA_train.sample(n=20000)
    #sample2 = CCA_test.sample(n=6000)

    y_train = sample1[target]
    y_test = sample2[target]

    print("Dropping targets")
    #X_train = training_data.drop(targets, axis = 1)
    #X_test = test_data.drop(targets, axis = 1)

    X_train = sample1.drop(targets, axis = 1)
    X_test = sample2.drop(targets, axis = 1)

    print("Creating and applying scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Creating classifier")
    #svm_classifier = SVC(kernel="rbf", C=1.0, gamma="auto")

    print("Fitting classifier")
    #svm_classifier.fit(X_train_scaled, y_train)

    print("Predicting...")
    #y_pred = svm_classifier.predict(X_test_scaled)

    grid_search = GridSearchCV(SVC(), param_grid, scoring='f1_macro', cv=5)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    print("Best parameters: ", grid_search.best_params_)
    print("Best f1 score:", grid_search.best_score_)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

