from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from models.CCA import *
from models.RFE import *

def SVM(training_data, test_data, validate_data, target):
    print("initializing targets")

    param_grid = {
        'C': [ 1, 10, 100, 1000],
        'kernel':['rbf']
    }

    #targets = ["srcip","dstip","attack_cat", "Label"]
    targets = ["attack_cat", "Label"]
    
    #CCA_train, CCA_test, CCA_validate = correlation_coefficient(training_data, test_data, validate_data, target)
    train, test, validate = rfe_model(training_data, test_data, validate_data, target)

    sample1 = train.sample(n=10000)
    sample2 = validate.sample(n=5000)

    y_train = sample1[target]
    y_test = sample2[target]

    print("Dropping targets from X dataframe...")

    X_train = sample1.drop(targets, axis = 1)
    X_test = sample2.drop(targets, axis = 1)

    print("Creating and applying scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Grid searching for best parameters...")
    grid_search = GridSearchCV(SVC(), param_grid, scoring='f1_macro', cv=5)
    grid_search.fit(X_train_scaled, y_train)

    print("Predicting with best model... ")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    print("Best parameters: ", grid_search.best_params_)
    print("Best f1 score:", grid_search.best_score_)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

