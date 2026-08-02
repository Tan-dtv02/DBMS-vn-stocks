import joblib
from lightgbm import LGBMRegressor

from .config import RANDOM_STATE, MODEL_PATH, MODEL_DIR
from .utils import ensure_dirs


def train_lgbm_model(X_train, y_train):
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    ensure_dirs(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)

    return model