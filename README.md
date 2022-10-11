# Interactive MNB - CS593-HAI Assignment1
Yufan Chen(chen4076@purdue.edu)

## Setup
I use python flask to write the server side. The following libraries are needed.
```bash
# Lib Version 
Python 3.9.7
Flask 2.2.2
Werkzeug 2.2.2
nltk 3.7
numpy 1.22.1
pandas 1.2.4
scikit-learn 0.24.2
```

## Run
After installing these libraries, simply run ```flask run -p 5000``` under the directory of ```app.py``` file.

Then the Multinomial Naive Bayes classifier will begin to train and conduct cross-validataion on the ```dbpedia_8K.csv``` dataset in the ```dataset``` directory. 

Wait about 10 minutes until the classifier finishes training and evaluations, then server will be ready.

The web application is rendered on ```https://localhost:5000```.

If you need to render on another port, you should change the form action address accordingly in the ```/templates/index.html``` file.

## Results
The mean accuracy on ```dbpedia_8K.csv``` dataset is 0.8547499999999999, which is computed by the 10-fold validation.

On the web application, the users can input custom text to predict its category. The important words that contribute most to the prediction will be displayed in the barchart. The y axis value is the exactly raw importance value P(w|c).

$$P(w|c) = \frac{\alpha_{wc} + F_{wc}}{\sum^N_{x=1}\alpha_{xc} + \sum^N_{x=1}F_{xc}}$$

 They can manually add a word to this chart. Once a word is added, its importance will be calculated and display.

 They can manually remove a word from the prediction in particular category.

 They can also manually adjust the word's importance factor, that is the smoothing factor $\alpha$ in the P(w|c) formula, by using the adjust word importance feature. They can indicate which word to adjust and furtherly they can indicate a category other than the predicted category.