import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def test_lightgbm_pipeline_smoke(tmp_path: object) -> None:
    train_df = pd.DataFrame(
        {
            "text": [
                "account login issue",
                "billing charge problem",
                "password reset request",
                "payment failure error",
            ],
            "label": ["auth", "billing", "auth", "billing"],
        }
    )

    X = train_df["text"].values
    y = train_df["label"].values

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100)),
            ("clf", LGBMClassifier(random_state=42, n_estimators=2)),
        ]
    )
    pipeline.fit(X, y)

    probas = pipeline.predict_proba(["login problem"])
    assert probas.shape[0] == 1
    assert probas.shape[1] == 2
