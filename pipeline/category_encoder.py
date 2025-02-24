from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


class category_encoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=15):
        self.label_encoder = LabelEncoder()
        self.target_encoder = ce.TargetEncoder(smoothing=smoothing)
        self.marca_means = None
        self.global_mean = None
        self.known_makes = None
        self.known_models = None
        self.cat_cols = None

    def fit(self, X, y):
        if 'trasmissione' in X.columns:
            self.label_encoder.fit(X['trasmissione'])

        self.cat_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns if
                         col != 'trasmissione']

        self.target_encoder.fit(X[self.cat_cols], y)

        self.marca_means = X.join(y.rename('target')).groupby('marca')['target'].mean()
        self.global_mean = y.mean()
        self.known_makes = set(X['marca'])
        self.known_models = set(X['modello'])

        return self

    def transform(self, X):
        X = X.copy()

        if 'trasmissione' in X.columns:
            X['trasmissione'] = self.label_encoder.transform(X['trasmissione'])

        encoded = self.target_encoder.transform(X[self.cat_cols])

        new_models_mask = ~X['modello'].isin(self.known_models)
        if new_models_mask.any():
            replacements = X.loc[new_models_mask, 'marca'].map(self.marca_means).fillna(self.global_mean)
            encoded.loc[new_models_mask, 'modello'] = replacements.values

        X[self.cat_cols] = encoded
        return X
