from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_models(random_state: int = 42):
    """
    Return supervised binary classification models for next-basket prediction.
    """

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),

        "knn": KNeighborsClassifier(
            n_neighbors=25,
        ),

        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=random_state,
        ),

        # "rbf_svm": SVC(
	#     kernel="rbf",
	#     probability=True,
	#     class_weight="balanced",
	#     random_state=random_state,
	# ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=64,
            class_weight="balanced",
            random_state=random_state,
        ),

        "catboost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_state=random_state,
        ),
    }

    return models