from sklearn.pipeline import Pipeline


def predict(pipeline_fitted: Pipeline, text: str):
    return pipeline_fitted.predict_proba(text)
