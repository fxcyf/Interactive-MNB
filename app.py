import json
from flask import Flask, redirect, url_for, request, render_template
from MNB import MNB
import pandas as pd

app = Flask(__name__)
model = MNB()

print("Setting up text classifier...")
df = pd.read_csv('dataset/dbpedia_8K.csv')

category_map = {0: 'Company',
                1: 'Education Institution',
                2: 'Artist',
                3: 'Athlete',
                4: 'Office Holder',
                5: 'Mean of Transportation',
                6: 'Building',
                7: 'Natural Place'}

pred_test_res = model.cross_validation(df)
user_input = ""
pred_user_res = {}

###### DEBUG ######
# pred_test_res = {'accuracy': 0.88}
# user_input = "text"
# pred_user_res = {
#     'cat_pred': 3,
#     'confidence': 0.99,
#     'important_words': [{'word': 'flow', 'importance': 0.008}, {'word': 'up', 'importance': 0.006}, {'word': 'down', 'importance': 0.004}, {'word': 'yeff', 'importance': 0.001}, {'word': 'ttt', 'importance': 0.0004}, {'word': 'flodfs', 'importance': 0.0001}, ]
# }
###### END DEBUG ######


@app.route('/')
def home():
    return render_template('index.html', accuracy=pred_test_res['accuracy'])


@app.route('/predict', methods=['POST'])
def predict():
    global user_input, pred_user_res
    user_input = request.form['text']
    pred_user_res = model.predict_user(user_input)
    return render_data()


@app.route('/add', methods=['POST'])
def add_word():
    new_word = request.form['word']
    category = request.form['category']
    model.add_new_word(new_word, int(category))

    global pred_user_res
    pred_user_res = model.predict_user(user_input)

    return render_data()


@ app.route('/remove', methods=['POST'])
def remove_word():
    word = request.form['word']
    category = request.form['category']
    model.remove_word(word, int(category))

    global pred_user_res, pred_test_res
    pred_test_res = model.predict_test()
    pred_user_res = model.predict_user(user_input)
    return render_data()


@ app.route('/adjust-importance', methods=['POST'])
def adjust_word_importance():
    word = request.form['word']
    category = request.form['category']
    importance = request.form['importance']
    model.adjust_word_importance(word, int(category), float(importance))

    global pred_user_res, pred_test_res
    pred_test_res = model.predict_test()
    pred_user_res = model.predict_user(user_input)

    return render_data()


def render_data():
    return render_template('index.html',
                           accuracy=pred_test_res['accuracy'],
                           cat_pred=pred_user_res['cat_pred'],
                           confidence="{0:.2f}%".format(
                               pred_user_res['confidence'] * 100),
                           important_words=json.dumps(
                               pred_user_res['important_words']),
                           text=user_input,
                           categories=category_map,
                           words=list(map(lambda d: d['word'],
                                          pred_user_res['important_words'])))


if __name__ == '__main__':
    app.run()
