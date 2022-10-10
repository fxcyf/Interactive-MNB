from unicodedata import category
from flask import Flask, redirect, url_for, request, render_template
from MNB import MNB
import pandas as pd

app = Flask(__name__)
model = MNB()

print("Setting up text classifier...")
df = pd.read_csv('dbpedia_8K.csv')

random_state = 25
train_data = df.sample(frac=0.8, random_state=random_state)
test_data = df.drop(train_data.index)

model.train(train_data)
pred_test_res = model.predict_test(test_data)

user_input = ""
pred_user_res = {}

category_map = {0: 'Company',
                1: 'Education Institution',
                2: 'Artist',
                3: 'Athlete',
                4: 'Office Holder',
                5: 'Mean of Transportation',
                6: 'Building',
                7: 'Natural Place'}


@app.route('/')
def home():
    return render_template('index.html', accuracy=pred_test_res['accuracy'], predict_res=1)


@app.route('/predict', methods=['POST'])
def predict():
    global user_input, pred_user_res
    user_input = request.form['text']
    pred_user_res = model.predict_user(user_input)
    return render_template('index.html',
                           accuracy=pred_test_res['accuracy'],
                           cat_pred=category_map[pred_user_res['cat_pred']],
                           confidence="({0:.2f}%)".format(
                               pred_user_res['confidence'] * 100),
                           important_words=pred_user_res['important_words'],
                           text=user_input,
                           categories=category_map
                           )


@app.route('/add', methods=['POST'])
def add_word():
    new_word = request.form['word']
    category = request.form['category']
    model.add_new_word(new_word, int(category))

    global pred_user_res
    pred_user_res = model.predict_user(user_input)

    return render_template('index.html',
                           accuracy=pred_test_res['accuracy'],
                           cat_pred=category_map[pred_user_res['cat_pred']],
                           confidence="({0:.2f}%)".format(
                               pred_user_res['confidence'] * 100),
                           important_words=pred_user_res['important_words'],
                           text=user_input,
                           categories=category_map)


@app.route('/remove', methods=['POST'])
def remove_word():
    word = request.form['word']
    category = request.form['category']
    model.remove_word(word, int(category))
    global pred_user_res, pred_test_res
    pred_test_res = model.predict_test(test_data)
    pred_user_res = model.predict_user(user_input)

    return render_template('index.html',
                           accuracy=pred_test_res['accuracy'],
                           cat_pred=category_map[pred_user_res['cat_pred']],
                           confidence="({0:.2f}%)".format(
                               pred_user_res['confidence'] * 100),
                           important_words=pred_user_res['important_words'],
                           text=user_input,
                           categories=category_map)


@app.route('/adjust-importance', methods=['POST'])
def adjust_word_importance():
    word = request.form['word']
    category = request.form['category']
    importance = request.form['importance']
    model.adjust_word_importance(word, int(category), float(importance))

    pred_test_res = model.predict_test(test_data)
    pred_user_res = model.predict_user(user_input)

    return render_template('index.html',
                           accuracy=pred_test_res['accuracy'],
                           cat_pred=category_map[pred_user_res['cat_pred']],
                           confidence="({0:.2f}%)".format(
                               pred_user_res['confidence'] * 100),
                           important_words=pred_user_res['important_words'],
                           text=user_input,
                           categories=category_map)


if __name__ == '__main__':
    app.run(debug=True)
