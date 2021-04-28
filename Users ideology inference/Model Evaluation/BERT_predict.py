import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import csv
# BERT Eval
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
import pickle
from collections import Counter


'''
Making Predictions on unseen data
'''

def predict(dataloader_val, model, device): # NEW evaluate

    model.eval()
    predictions = list()
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        batch_input = batch[0]

        with torch.no_grad():        
            outputs = model(batch_input)

        logits = outputs['logits']   
        logits = torch.sigmoid(logits)

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def TestSet_Evaluation(df, model_path):
    # setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    Data Preparation
    '''
    # BertTokenizer: tokenizing texts and turning them into integers vectors
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
    # encoding tokenized texts to indexes
    encoded_data_val = tokenizer.batch_encode_plus(
    df.clean_text.values, 
    add_special_tokens=True, 
    return_attention_mask=False, 
    padding=True, 
    truncation=True, 
    return_tensors='pt'
    )
    input_ids_val = encoded_data_val['input_ids']
    #labels_val = torch.tensor(df.label.values)
    dataset_val = TensorDataset(input_ids_val)
    print('end tokenization and encoding')
    batch_size = 4
    # iterable DatLoader
    dataloader_inference = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)
    '''
    Loading and Evaluating Model
    '''
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=1,
                                                      output_attentions=False,
                                                      output_hidden_states=False)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    #model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    predictions = predict(dataloader_inference, model, device)
    return predictions


# Load the dataset into a pandas dataframe.
df = pd.read_csv('guncontrol_cleanedpost.csv')
model_path = 'models/BERT/with_punt/finetuned_BERT_epoch_1.model'
# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))
TestSet_Evaluation(df, model_path)