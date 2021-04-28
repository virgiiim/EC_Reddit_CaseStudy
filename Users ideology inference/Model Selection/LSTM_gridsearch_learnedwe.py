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

# Others
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


#Loading cleaned training dataset
training_set = pd.read_csv('ground_truth_datasets/cleaned_TrainingSet_LSTM.csv')
# features for LSTM
t_features = training_set['clean_text']
labels = training_set['pol_leaning'] 
# n_records used for training
print(labels.value_counts())

### Tokenize text and Create sequence for LSTM
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(t_features) #This method creates the vocabulary index based on word frequency (0 is reserved for padding)
sequences = tokenizer.texts_to_sequences(t_features) #This method transforms each text in texts to a sequence of integers (the index created before)
data = pad_sequences(sequences, maxlen=350) #This method pads sequences to the same lenght (60)

num_types=len(tokenizer.word_index) #number of types in input dataset

# Function to create model, required for KerasClassifier
def create_model(output_dim=100,units=32):
  # create model
    model = Sequential()
    model.add(Embedding(num_types+1, output_dim=output_dim, input_length=350, mask_zero=True))
    model.add(LSTM(units=units, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# split into input (X) and output (Y) variables
X = data
Y = labels

# create model
model = KerasClassifier(build_fn=create_model, epochs=10 , verbose=2)

# patient early stopping
es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=2)

# define the grid search parameters
output_dim = [100,300] 
units = [32,64,128]
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

with open('./grid_result_LearnedWE.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train the best model obtained through GridSearch
LSTM_units = grid_result.best_params_['units']
emb_dim = grid_result.best_params_['output_dim']
# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
# Network architecture
model = Sequential()
model.add(Embedding(num_types+1, emb_dim, input_length=350, mask_zero=True))
model.add(LSTM(LSTM_units, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(data, np.array(labels), validation_split=0.2, epochs=10, shuffle=True ,callbacks=[es])
# Saving history
with open('./history_bestmodel_LearnedWE.pickle', 'wb') as handle:
    pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Saving best model, weights and tokenizer
# serialize model to JSON
model_json = model.to_json()
with open("models/LSTM/best_modelLEARNED.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/LSTM/best_modelLEARNED_weights.h5")
# saving tokenizer
with open('models/LSTM/tokenizer_learned.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved model to disk")