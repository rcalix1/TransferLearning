from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

## to fine-tune, put it in training mode
model.train()

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('bert-base-uncased')

##################################

from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-3)

###############################

# one step forward
import torch

texts= [    "this is a good example",
            "this is a bad example",
            "this is a good one"         ]

labels= [1,0,1]


labels = torch.tensor(labels).unsqueeze(0)

####################################################

encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

####################################################

input_ids      = encoding['input_ids']

attention_mask = encoding['attention_mask']

#####################################################

outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

loss = outputs.loss

loss.backward()     ## back prop

optimizer.step()

#####################################################

print(  outputs   )

#####################################################

#Manually calculate loss

from torch.nn import functional

labels = torch.tensor([1,0,1])

outputs = model(input_ids, attention_mask=attention_mask)

loss = functional.cross_entropy(outputs.logits, labels)

loss.backward()

optimizer.step()

print(loss)

print(outputs)


#######################################

from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

############################################

from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

from transformers import BertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

############################################

import datasets

from datasets import load_dataset

sst2= load_dataset("glue","sst2")

from datasets import load_metric

metric = load_metric("glue", "sst2")

texts   = sst2['train']['sentence']
labels  = sst2['train']['label']

val_texts  = sst2['validation']['sentence']
val_labels = sst2['validation']['label']

len(texts)

#############################################

# I will take small portion

K=10000

train_dataset =  MyDataset(tokenizer(texts[:K], truncation=True, padding=True), labels[:K])
val_dataset   =  MyDataset(tokenizer(val_texts, truncation=True, padding=True), val_labels)

#############################################

from torch.utils.data import DataLoader
from transformers import  AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

train_loader =  DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   =  DataLoader(val_dataset, batch_size=16, shuffle=True)

##############################################

optimizer = AdamW(model.parameters(), lr=1e-5)

##############################################

for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()


    model.eval()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        predictions=outputs.logits.argmax(dim=-1)  
        metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

    eval_metric = metric.compute()
    print(f"epoch {epoch}: {eval_metric}")


###################################



