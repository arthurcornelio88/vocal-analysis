"""Classificador XGBoost para mecanismos laríngeos M1/M2."""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_mechanism_classifier(
    df_features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "mechanism_label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Treina classificador XGBoost para diferenciar M1/M2.

    Args:
        df_features: DataFrame com features e labels.
        feature_cols: Lista de colunas de features. Default: ['f0', 'hnr', 'energy'].
        target_col: Nome da coluna target. 0=M1/Peito, 1=M2/Cabeça.
        test_size: Proporção do conjunto de teste.
        random_state: Seed para reprodutibilidade.

    Returns:
        Modelo XGBoost treinado.
    """
    if feature_cols is None:
        feature_cols = ["f0", "hnr", "energy"]

    X = df_features[feature_cols]
    y = df_features[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective="binary:logistic",
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    print("Relatório de Classificação M1/M2:")
    print(classification_report(y_test, model.predict(X_test)))

    return model
