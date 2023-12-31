import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class Split():
    '''

    INPUT:

        Data path,

        n_split = (

            2: 50% - 50% (Train - test),

            3: 60% - 40%,

            5: 80% - 20%

        )



    OUTPUT: 80-10-10 TRAIN-DEV-TEST

        X_train, y_train, X_val, y_val, X_test, y_test

    '''

    def __init__(self, data_path, n_split):
        self.data_path = data_path

        self.n_split = n_split

    def preprocess_data_format(self):
        labels = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']

        data = pd.read_csv(self.data_path)

        X = data.drop(columns=labels)

        y = data[labels]

        X = X.to_numpy()

        y = y.to_numpy()

        return X, y

    def count_label_distribution(self, y):
        label_counts = pd.DataFrame(data=y,
                                    columns=['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam'])

        label_distribution = label_counts.apply(lambda x: x.value_counts()).fillna(0).astype(int)

        return label_distribution

    def split(self, X, y):
        mskf = MultilabelStratifiedKFold(n_splits=self.n_split, shuffle=True, random_state=0)

        for train_index, test_index in mskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_distribution(self):
        X, y = self.preprocess_data_format()

        X_train, y_train, X_val, y_val, X_test, y_test = self.split(X, y)

        train_label_distribution = self.count_label_distribution(y_train)

        dev_label_distribution = self.count_label_distribution(y_val)

        test_label_distribution = self.count_label_distribution(y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test, train_label_distribution, dev_label_distribution, test_label_distribution

    def run(self):
        X, y = self.preprocess_data_format()

        X_train, y_train, X_val, y_val, X_test, y_test = self.split(X, y)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def save_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):
        base_path = "C:\\Users\\broke\\PycharmProjects\\Smart-Sentiment-Analysis\\data\\preprocessed_data\\"

        train_data = np.column_stack((X_train, y_train))
        val_data = np.column_stack((X_val, y_val))
        test_data = np.column_stack((X_test, y_test))

        columns = ['Review', 'giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']

        pd.DataFrame(train_data, columns=columns).to_csv(base_path + 'train_data.csv', index=False)
        pd.DataFrame(val_data, columns=columns).to_csv(base_path + 'val_data.csv', index=False)
        pd.DataFrame(test_data, columns=columns).to_csv(base_path + 'test_data.csv', index=False)


if __name__ == '__main__':
    DATA_PATH = r"C:\Users\broke\PycharmProjects\Smart-Sentiment-Analysis\data\raw_data\data_final_problem2.csv"
    split = Split(DATA_PATH, 5)

    X_train, y_train, X_val, y_val, X_test, y_test, train_label_dist, dev_label_dist, test_label_dist = split.test_distribution()

    print("Training Label Distribution:")
    print(train_label_dist)

    print("Development Label Distribution:")
    print(dev_label_dist)

    print("Testing Label Distribution:")
    print(test_label_dist)

    # Save the processed data
    split.save_datasets(X_train, y_train, X_val, y_val, X_test, y_test)