import os
import joblib
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from data_preparation import data_preparation
from pre_process import pre_process
from model_trainer import model_trainer


class pipeline:
    def __init__(self, n_splits=10, use_kfold=True):
        self.output_dir = '../diagrams/regression'
        self.data_preparator = data_preparation()
        self.pre_processor = pre_process()
        self.n_splits = n_splits
        self.use_kfold = use_kfold
        self.trainer = model_trainer(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)


    def prepare_data(self, data_path):
        processed_path = os.path.join(os.path.dirname(data_path), "processed_data.csv")

        if os.path.exists(processed_path):
            print(f"Trovato dataset pre-processato: {processed_path}")
            df = pd.read_csv(processed_path, on_bad_lines='skip')
        else:
            print("Lettura dataset originale...")
            df = pd.read_csv(data_path, on_bad_lines='skip')

            print("Esecuzione pre-processing...")
            processed_path = self.pre_processor.preprocess(df)
            print(f"Pre-processing completato e salvato in {processed_path}")

            df = pd.read_csv(processed_path, on_bad_lines='skip')

        X = df.drop(columns=['prezzo'])
        y = df['prezzo']
        return X, y

    def run_pipeline(self, path):
        X, y = self.prepare_data(path)

        if self.use_kfold:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            for fold, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                print(f'\nData preparation per il fold {fold + 1}/{self.n_splits}...')
                X_train = self.data_preparator.fit(X_train, y_train)
                X_train = self.data_preparator.transform_train(X_train)
                X_test = self.data_preparator.transform_test(X_test)

                y_train = y_train.loc[X_train.index]
                y_test = y_test.loc[X_test.index]

                self.trainer.train(X_train, X_test, y_train, y_test, fold)
        else:
            print('\nTraining su tutto il dataset senza K-Fold...')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train = self.data_preparator.fit(X_train, y_train)
            X_train = self.data_preparator.transform_train(X_train)
            X_test = self.data_preparator.transform_test(X_test)

            # Filtrare y_train dopo la trasformazione
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]

            self.trainer.train(X_train, X_test, y_train, y_test, fold=None)

        print('Training completato. Salvataggio risultati...')
        if self.use_kfold:
            self.trainer.save_results()

        final_results = pd.DataFrame(self.trainer.results)
        print('\nRisultati finali su tutti i fold:')
        print(final_results)

        joblib.dump(self.trainer.model, 'random_forest_regressor_model.pkl')
        joblib.dump({'preparator': self.data_preparator}, 'pipeline_regressor_transformers.pkl')


if __name__ == "__main__":
    pipe = pipeline(n_splits=10, use_kfold=False)
    path = "../datasets/car_prices.csv"
    pipe.run_pipeline(path)
