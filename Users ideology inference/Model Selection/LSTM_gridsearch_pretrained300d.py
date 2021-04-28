import tensorflow as tf
# Keras
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

#Scikit-learn 
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

#Nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Others
import pickle
import numpy as np
import pandas as pd
import re
import stop_words
from stop_words import get_stop_words
stop_words = get_stop_words('en')


#Loading cleaned training dataset
training_set = pd.read_csv('ground_truth_datasets/cleaned_TrainingSet_LSTM.csv')
# features for LSTM
t_features = training_set['clean_text']
labels = training_set['pol_leaning'] 
# n_records used for training
print(labels.value_counts())

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(t_features)

# integer encode the documents
encoded_docs = t.texts_to_sequences(t_features)

# pad documents to a max length of 60 words (mean length is about 95)
padded_docs = pad_sequences(encoded_docs, maxlen=350, padding='post')
vocab_size=len(t.word_index) #number of types in input dataset

#Loading embeddings
embeddings_index = dict()
f = open('glove_embeddings/glove.6B.300d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#Creating embedding matrix
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Function to create model, required for KerasClassifier
def create_model(output_dim=300,units=32):
  # create model
    model = Sequential()
    model.add(e)
    model.add(LSTM(units=units, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# split into input (X) and output (Y) variables
X = padded_docs
Y = labels

# create model
model = KerasClassifier(build_fn=create_model, epochs=10 , verbose=2)

# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=350, trainable=True) #Trainable = True to update weights during training

# define the grid search parameters
output_dim=[300] 
units=[32,64,128]
param_grid = dict(output_dim=output_dim,  units=units )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4, cv=3, verbose=5)
grid_result = grid.fit(X, Y, callbacks=[es])

# summarize results
res = {'mean:':[], 'stdev':[], 'params':[]}
# =============================================================================
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# =============================================================================
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    res['mean'].append(mean)
    res['stdev'].append(stdev)
    res['params'].append(param)
# =============================================================================
    print("%f (%f) with: %r" % (mean, stdev, param))
# =============================================================================

with open('./grid_result_PretrainedWE300.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train the best model obtained through GridSearch
LSTM_units = grid_result.best_params_['units']
emb_dim = grid_result.best_params_['output_dim']
# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
# Network architecture
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=350, trainable=True) #Trainable = True to update weights during training
## Network architecture
model = Sequential()
model.add(e)
model.add(LSTM(LSTM_units, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(padded_docs, labels, validation_split = 0.2 ,epochs=10, shuffle=True, )
# Saving history
with open('./history_bestmodel_PretrainedWE300.pickle', 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Saving best model, weights and tokenizer
# serialize model to JSON
model_json = model.to_json()
with open("models/LSTM/best_modelPretrained300.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/LSTM/best_modelPretrained300_weights.h5")
# saving tokenizer
with open('models/LSTM/tokenizer_Pretrained300.pickle', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved model to disk")