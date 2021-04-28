import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn  

''' 
    Data Preparation
'''
df = pd.read_csv('ground_truth_datasets/PUNT_cleaned_TrainingSet_LSTM.csv')
# dropping unecessary columns for classification
df.drop(columns=['text', 'tokenized', 'length'], inplace=True)
print('n_articles for each label:', df['pol_leaning'].value_counts())
# defining Training and Validation Split
X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.pol_leaning.values, 
                                                  test_size=0.10, 
                                                  random_state=42, 
                                                  stratify=df.pol_leaning.values)

df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
print(df.groupby(['pol_leaning', 'data_type']).count())

'''
    Tokenizing and Encoding Data
'''
# BertTokenizer: tokenizing texts and turning them into integers vectors
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
                                          
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].clean_text.values, 
    add_special_tokens=True, 
    return_attention_mask=False, 
    padding=True,
    truncation=True, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].clean_text.values, 
    add_special_tokens=True, 
    return_attention_mask=False,
    padding=True, 
    truncation=True, 
    return_tensors='pt'
)
# encoding training
input_ids_train = encoded_data_train['input_ids']
labels_train = torch.tensor(df[df.data_type=='train'].pol_leaning.values.astype(float)).view(-1,1)
# encoding validation
input_ids_val = encoded_data_val['input_ids']
labels_val = torch.tensor(df[df.data_type=='val'].pol_leaning.values.astype(float)).view(-1,1)
# building encoded dataset for both training and validation
dataset_train = TensorDataset(input_ids_train, labels_train)
dataset_val = TensorDataset(input_ids_val, labels_val)
print('end tokenization and encoding')

'''
   Uploading BERT Model and defining parameters
'''
# BERT Pre-trained Model: bert-base-uncased is a smaller pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=1,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Data Loaders: DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset
batch_size = 4
dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)
dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)  
# Init CrossEntropy Loss
criterion = nn.BCEWithLogitsLoss()                   
epochs = 4
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

'''
    Compute Performance Metrics (F1 Score, Accuracy)
'''
# Performance Metrics: f1 score, Accuracy
def f1_score_func(preds, labels):
    preds_new = np.where(preds.flatten()>0.5, 1,0).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_new, pos_label=1)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    preds_new = np.where(preds.flatten()>0.5, 1, 0)
    labels_flat = labels.flatten()
    return np.sum(preds_new == labels_flat) / len(labels_flat)

'''
    Training Loop
'''
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    total_eval_accuracy = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        batch_input = batch[0]
        batch_target = batch[1]

        with torch.no_grad():        
            outputs = model(batch_input)

        logits = outputs['logits']   
        loss = criterion(logits, batch_target) 
        loss_val_total += loss.item()

        logits = torch.sigmoid(logits)

        logits = logits.detach().cpu().numpy()
        label_ids = batch_target.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(dataloader_val)
    return loss_val_avg, predictions, true_vals, avg_val_accuracy

for epoch in range(1, epochs+1):
    
    model.train()
    
    loss_train_total = 0

    for batch in dataloader_train:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)

        batch_input = batch[0]
        batch_target = batch[1]

        outputs = model(batch_input)
        logits = outputs['logits']

        loss = criterion(logits, batch_target) 
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
         
        
    torch.save(model.state_dict(), f'models/BERT/punt2/finetuned_BERT_epoch_{epoch}.model')
    # recording performances from this epoch
    print(f'\nEpoch {epoch}')
    loss_train_avg = loss_train_total/len(dataloader_train)            
    print(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals, avg_val_accuracy = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    print(f'Validation Accuracy: {avg_val_accuracy}')
    print(f'Validation Loss: {val_loss}')
    print(f'Validation F1 Score: {val_f1}')
    training_stats.append(
        {
            'epoch': epoch,
            'Training Loss': loss_train_avg,
            'Valid. Loss': val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Vaid. F1 Score': val_f1,
        }
    )

'''
    Saving and plotting performances across all epochs
'''
# creating a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
print(df_stats)
# plot Learning Curve
sns.set(style='darkgrid')
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
plt.title("BERT Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])
plt.savefig('graphs/BERT_PUNT2_learningcurve.png') 
