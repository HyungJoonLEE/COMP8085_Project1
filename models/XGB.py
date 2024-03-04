import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score , classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def xgb_model(training_set, test_set):
    # Load datasets
    xgb_train = training_set
    xgb_test = test_set

