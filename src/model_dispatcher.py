# machine learning
from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import xgboost as xgb



models={
        "knn":KNeighborsClassifier(),
        # "mnb":MultinomialNB(),
        "log_reg":LogisticRegression(random_state=42),
        "sgd":SGDClassifier(random_state=42,loss='log'),
        "svm":SVC(probability=True, random_state=42),
        "dt":DecisionTreeClassifier(random_state=42),
        "rf":RandomForestClassifier(random_state=42),
        "gb":GradientBoostingClassifier(random_state=42),
        "nn":MLPClassifier(random_state=42),
        "xgb":xgb.XGBClassifier(random_state=42)
        }