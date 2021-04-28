from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import pandas as pd
import json
import re
import stop_words
import numpy as np
from stop_words import get_stop_words
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def remove_stopWords(s):
    stop_words = get_stop_words('en')
    s = ' '.join(word for word in s.split() if word not in stop_words)
    return s

class LSTModel():

    def __init__(self, file_model, file_weights, file_tokenizer):
        #loading model and weights
        json_file = open(file_model,  'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(file_weights)
        #loading tokenizer
        with open(file_tokenizer,'rb') as handle:
          self.tokenizer = pickle.load(handle)

    def predict_prob(self, submissions):
        #convert into lowercase
        submissions = submissions.apply(lambda x : str.lower(x))
        #remove punctuation and numbers
        submissions = submissions.apply(lambda x : " ".join(re.findall('[\w]+',x)))
        #applying tokenizer to test data
        encoded_docs_test = self.tokenizer.texts_to_sequences(submissions)
        padded_docs_test = pad_sequences(encoded_docs_test,maxlen=350, padding='post')
        return self.model.predict_proba(padded_docs_test)
    
    def predict_class(self, submissions):
        pred_prob = self.predict_prob(submissions)
        return np.round(pred_prob)

if __name__ == '__main__':
    ground_truth_data = pd.read_csv(r'C:\Users\virgi\Desktop\ASSEGNO\Paper_ec\ground_truth_datasets\NOPUNT_author_groundtruth.csv')
    file_model = r'C:\Users\virgi\Desktop\ASSEGNO\Paper_ec\Model Selection\Models\LSTM\LSTM_model_glove.json'
    file_weights = r'C:\Users\virgi\Desktop\ASSEGNO\Paper_ec\Model Selection\Models\LSTM\LSTM_model_glove.h5'
    file_tokenizer = r'C:\Users\virgi\Desktop\ASSEGNO\Paper_ec\Model Selection\Models\LSTM\LSTM_tokenizer_def.pickle'
    model = LSTModel(file_model, file_weights, file_tokenizer)
    pred_prob = model.predict_prob(ground_truth_data['text'])
    pred_label = model.predict_class(ground_truth_data['text']) #elimino questa parte per tutti gli utenti perchè non ho base su cui validare
    true_label = ground_truth_data['pol_leaning'] 
    # computing performances
    actual = true_label
    predicted = pred_label[0]
    cf= confusion_matrix(actual, predicted)
    print ('Confusion Matrix :')
    print(cf)
    print ('Accuracy Score :',accuracy_score(actual, predicted))
    print('Classification Report : ')
    print (classification_report(actual, predicted))
