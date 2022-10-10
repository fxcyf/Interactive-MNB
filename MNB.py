from operator import itemgetter
from sklearn import metrics
from math import ceil, log
import time
import re
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('omw-1.4')


class MNB:

    def __init__(self) -> None:
        self.cat_num_docs = {}  # {category: num of documents}

        # {category: -log(num of doc / total num of doc)}
        self.cat_neg_log_prob = {}

        self.cat_word_dict = {}  # {category_i: {word_j: count_ij}}
        self.cat_word_count_dict = {}  # {category_i: count_i}
        self.cat_word_smooth_dict = {}  # {category_i: {word_j: smooth term_ij}}
        # {category_i: {word_j: P(w_j|c_i)}}
        self.cat_word_importance_dict = {}

        self.vocab_len = 0  # number of uniqle words in all documents
        self.lemmatizer = WordNetLemmatizer()

        self.cat_udf_important_words = {}
        self.cat_udf_remove_words = {}

    def get_list_words(self, text):

        return [self.lemmatizer.lemmatize(w.lower()) for w in text if w.isalpha() and
                len(w) > 1 and
                w.lower() not in set(stopwords.words('english'))]

    def cross_validation(self, data):
        data.sample(frac=1)
        fragment_size = ceil(len(data) / 10)
        test_accuracy = []
        print(fragment_size)
        for i in range(10):
            test_data = data.iloc[i *
                                  fragment_size: min(len(data), (i + 1) * fragment_size)]
            train_data = data.drop(test_data.index)
            self.train(train_data)
            test_accuracy.append(self.predict_test(test_data)['accuracy'])
        print("Mean accuracy =", np.array(test_accuracy).mean())

    def train(self, train_data):
        print('Begin training classifier...')
        start_time = time.time()

        len_train_data = len(train_data)

        for i in range(8):
            self.cat_num_docs[i] = 0
        for label in train_data['label']:
            self.cat_num_docs[label] += 1
        for label in self.cat_num_docs:
            self.cat_neg_log_prob[label] = - \
                log(self.cat_num_docs[label] / len_train_data)

        word_set = set()

        for i, row in train_data.iterrows():
            text = re.split('\W+', (' '.join([row['title'], row['content']])))
            list_words = self.get_list_words(text)
            cat = row['label']

            word_set.update(list_words)
            self.cat_word_count_dict[cat] = self.cat_word_count_dict.get(
                cat, 0) + len(list_words)

            self.cat_word_dict[cat] = self.cat_word_dict.get(cat, {})
            for w in list_words:
                self.cat_word_dict[cat][w] = self.cat_word_dict[cat].get(
                    w, 0) + 1

        self.vocab_len = len(word_set)
        print("The Classifier is trained and it took {0:.2f} seconds".format(
            time.time() - start_time))

    def predict_test(self, test_data):
        print('Begin testing classifier...')
        li_results = []
        start_time = time.time()

        for i, row in test_data.iterrows():
            [cat_pred_log_prob_dict, _] = self.compute_prob(
                (' '.join([row['title'], row['content']])))
            cat_pred = max(cat_pred_log_prob_dict,
                           key=cat_pred_log_prob_dict.get)

            li_results.append((i, cat_pred, row['label']))
        print("The Classifier is tested and it took {0:.2f} seconds".format(
            time.time() - start_time))

        y_pred = [result[1] for result in li_results]
        y_true = [result[2] for result in li_results]
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("Accuracy = ", accuracy)
        return {'accuracy': accuracy}

    def predict_user(self, text):
        print('Begin predicting user input...')
        [cat_pred_log_prob_dict,
         cat_word_importance_dict] = self.compute_prob(
            text)  # P(w|c)

        cat_pred = max(cat_pred_log_prob_dict, key=cat_pred_log_prob_dict.get)
        prob_pred_sftmx = softmax(
            np.array(list(cat_pred_log_prob_dict.values())))
        confidence = np.max(prob_pred_sftmx)
        word_importance_dict = cat_word_importance_dict[cat_pred]
        top_10_words = dict(
            sorted(word_importance_dict.items(), key=itemgetter(1), reverse=True)[:10])

        udf_important_words = self.cat_udf_important_words.get(cat_pred, set())
        udf_important_words_dict = {
            w: self.cat_word_importance_dict[cat_pred][w] for w in udf_important_words}
        important_words = {**top_10_words, **udf_important_words_dict}

        print('cat pred: ', cat_pred)
        print('confidence: ', confidence)
        print('important_words: ', important_words)

        return {
            'cat_pred': cat_pred,
            'confidence': confidence,
            'important_words': important_words
        }

    def compute_prob(self, text):
        list_words = self.get_list_words(re.split('\W+', text))
        cat_pred_log_prob_dict = {}
        cat_word_importance_dict = {}  # P(w|c)
        for cat in self.cat_word_count_dict:
            neg_log_prob = self.cat_neg_log_prob[cat]
            word_dict = self.cat_word_dict[cat]
            count_cat = self.cat_word_count_dict[cat]
            word_smooth_dict = self.cat_word_smooth_dict.get(cat, {})
            cat_word_importance_dict[cat] = cat_word_importance_dict.get(cat, {
            })
            self.cat_word_importance_dict[cat] = self.cat_word_importance_dict.get(cat, {
            })
            udf_remove_words = self.cat_udf_remove_words.get(cat, set())

            for w in list_words:
                if w in udf_remove_words:
                    continue
                count_word_train = word_dict.get(w, 0)
                word_smooth = word_smooth_dict.get(
                    w, 1)  # The default smooth term is 1
                ratio = (count_word_train + word_smooth) / \
                    (count_cat + self.vocab_len)

                # word importance
                cat_word_importance_dict[cat][w] = ratio
                self.cat_word_importance_dict[cat][w] = ratio

                neg_log_prob -= log(ratio)

            cat_pred_log_prob_dict[cat] = -neg_log_prob
        return [cat_pred_log_prob_dict, cat_word_importance_dict]

    def add_new_word(self, word, cat):
        assert cat in self.cat_word_importance_dict
        word = self.lemmatizer.lemmatize(word.lower())
        self.cat_udf_important_words[cat] = self.cat_udf_important_words.get(
            cat, set())
        self.cat_udf_important_words[cat].add(word)

        if word in self.cat_udf_remove_words.get(cat, set()):
            self.cat_udf_remove_words[cat].remove(word)

        if word not in self.cat_word_importance_dict[cat]:
            word_dict = self.cat_word_dict[cat]
            count_word_train = word_dict.get(word, 0)

            count_cat = self.cat_word_count_dict[cat]

            word_smooth_dict = self.cat_word_smooth_dict.get(cat, {})
            word_smooth = word_smooth_dict.get(
                word, 1)  # The default smooth term is 1
            ratio = (count_word_train + word_smooth) / \
                (count_cat + self.vocab_len)
            self.cat_word_importance_dict[cat][word] = ratio

    def remove_word(self, word, cat):
        assert cat in self.cat_word_importance_dict
        word = self.lemmatizer.lemmatize(word.lower())

        self.cat_udf_remove_words[cat] = self.cat_udf_remove_words.get(
            cat, set())
        self.cat_udf_remove_words[cat].add(word)

        if word in self.cat_udf_important_words.get(cat, set()):
            self.cat_udf_important_words[cat].remove(word)

    def adjust_word_importance(self, word, cat, importance):
        assert importance > 0
        self.cat_word_smooth_dict[cat] = self.cat_word_smooth_dict.get(cat, {})

        old_importance = self.cat_word_smooth_dict[cat].get(word, 1)

        self.cat_word_smooth_dict[cat][word] = importance

        self.vocab_len += (importance - old_importance)


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)
    return f_x


def main():
    df = pd.read_csv('dbpedia_8K.csv')

    random_state = 25
    train_data = df.sample(frac=0.8, random_state=random_state)
    test_data = df.drop(train_data.index)

    model = MNB()

    # model.train(train_data)
    # model.predict_test(test_data)
    # model.predict_user("Daglish railway station is a commuter railway station on the boundary of Daglish and Subiaco, suburbs of Perth, Western Australia. Opened on 14 July 1924, the station was named after Henry Daglish, who was a mayor of Subiaco, a member for the electoral district of Subiaco, and a premier of Western Australia in the 1900s. The station consists of an island platform accessed by a pedestrian underpass. Daglish station is on the Fremantle line, and starting on 10 October 2022, the Airport line, which are both part of the Transperth network. Fremantle line services run every 10 minutes during peak hour and every 15 minutes outside peak hour and on weekends and public holidays. Upon the Airport line's opening, Fremantle line and Airport line services will run every 12 minutes during peak hour, for a combined frequency of a train every 6 minutes. The journey to Perth station is 4.9 kilometres (3.0 mi) long and takes 7 minutes.")
    model.cross_validation(df)


if __name__ == "__main__":
    main()
