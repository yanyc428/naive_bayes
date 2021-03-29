from data import load_data
from text import get_all_vocab, generate_vector
from segmentation import seg
import numpy as np
import pickle
import tqdm
import pandas as pd


class Model(object):
    def __init__(self, train_num=10, save=False, path=None):
        print("init model...")
        self.train_data_frame = load_data(train_num)
        print("generating vocab vector...")
        self.words = get_all_vocab(self.train_data_frame)
        print("getting classes...")
        self.classes = self.train_data_frame["class"].unique()
        print("getting word vector...")
        self.word_vec = list(set(self.words))
        print("getting word frequency...")
        self.word_freq = np.array([self.words.count(x) / len(self.words) for x in self.word_vec])
        print("get class probability")
        self.classes_prob = [1 if x in ' '.join(self.train_data_frame.loc[:, "content"].values) else 0 for x in
                             self.word_vec]
        print("calculating prior")
        self.prior = dict()
        for cls in tqdm.tqdm(self.classes):
            self.prior[cls] = np.array(self.get_prior(cls))
        print("finish init...")
        if path:
            if save:
                self.save(path)
                print("finish saving at " + path + "...")

    def get_word_freq(self):
        return [self.words.count(x) for x in self.word_vec]

    def get_prior(self, cls):
        df = self.train_data_frame.loc[self.train_data_frame["class"] == cls]
        vocab = get_all_vocab(df)
        # Laplace Smoothing
        return [(vocab.count(x) + 1) / (len(vocab) + len(self.classes)) for x in self.word_vec]

    def predict(self, content):
        vec = generate_vector(self.word_vec, seg(content))
        probability = []
        for cls in self.classes:
            likelihood = 1
            for index, x in enumerate(vec):
                if x > 0:
                    likelihood = likelihood * \
                                 (self.prior[cls][index] * (1 / len(self.classes)) / (
                                             self.classes_prob[index] / sum(self.classes_prob))) ** x

            probability.append(
                likelihood
            )
        return self.classes[np.array(probability).argmax()]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def evaluate(self, test_df):
        df = pd.DataFrame(columns=["class", "accuracy", "precision", "recall", "F1"])
        print("start evaluate...")
        test_df["predict"] = test_df["content"].apply(self.predict)
        print("start calculating")
        for cls in tqdm.tqdm(self.classes):
            real = test_df.loc[test_df["class"] == cls, "class"].values.tolist()
            predict = test_df.loc[test_df["class"] == cls, "predict"].values.tolist()
            real_n = test_df.loc[test_df["class"] != cls, "class"].values.tolist()
            predict_n = test_df.loc[test_df["class"] != cls, "predict"].values.tolist()
            print(cls, real, real_n)
            TP = predict.count(cls)
            TN = len(real) - predict.count(cls)
            FP = predict_n.count(cls)
            FN = len(real_n) - predict_n.count(cls)
            acc = (TP + FN) / (TP + TN + FP + FN)
            pre = (TP) / (TP + FP)
            rec = (TP) / (TP + TN)
            df = df.append(
                {"class": cls, "accuracy": acc, "precision": pre, "recall": rec, "F1": 2 * pre * rec / (pre + rec)},
                ignore_index=True)
        df.to_csv("result.csv")
        print("end evaluate...")
        return df


if __name__ == '__main__':
    # model = Model(100, True, "model_100.pkl")
    model = Model.load("model_100.pkl")
    print(model.evaluate(load_data(10)))
