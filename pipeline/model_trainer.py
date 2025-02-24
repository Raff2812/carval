from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os

class model_trainer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = []
        self.model = None
        os.makedirs(os.path.join(self.output_dir, 'feature_importances'), exist_ok=True)

    def train(self, X_train, X_test, y_train, y_test, fold):
        if fold is not None:
            print(f'\nTraining del modello con k-fold per fold {fold + 1}...')


        model = RandomForestRegressor(
            max_depth=20,
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            n_jobs=-1,
            random_state=20,
            bootstrap=True,
            criterion='squared_error'
        )

        model.fit(X_train, y_train)

        # Previsioni
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calcolo delle metriche
        fold_results = {
            'MAE_train': mean_absolute_error(y_train, y_train_pred),
            'MAE_test': mean_absolute_error(y_test, y_test_pred),
            'MSE_train': mean_squared_error(y_train, y_train_pred),
            'MSE_test': mean_squared_error(y_test, y_test_pred),
            'RMSE_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'RMSE_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAPE_train': mean_absolute_percentage_error(y_train, y_train_pred),
            'MAPE_test': mean_absolute_percentage_error(y_test, y_test_pred),
        }

        # Aggiunge il numero del fold solo se non è None
        if fold is not None:
            fold_results['Fold'] = fold + 1

        self.results.append(fold_results)
        self.model = model
        self._plot_feature_importance(model, X_train.columns, fold)

    def _plot_feature_importance(self, model, feature_names, fold=None):
        feature_importances = model.feature_importances_

        # Ordinare le feature per importanza
        sorted_indices = np.argsort(feature_importances)[::-1]  # Ordina in ordine decrescente
        sorted_features = feature_names[sorted_indices]
        sorted_importances = feature_importances[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, sorted_importances, color='skyblue')
        plt.xlabel('Importanza')
        plt.ylabel('Feature')

        if fold is not None:
            title = f'Feature Importance - Fold {fold + 1}'
            filename = f'{self.output_dir}/feature_importances/feature_importance_fold_{fold + 1}.png'
        else:
            title = 'Feature Importance - Full Training'
            filename = f'{self.output_dir}/feature_importances/feature_importance_full_training.png'

        plt.title(title)
        plt.savefig(filename)
        plt.close()

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        os.makedirs(f'{self.output_dir}/results', exist_ok=True)
        results_df.to_csv(f'{self.output_dir}/results/fold_results.csv', index=False)
