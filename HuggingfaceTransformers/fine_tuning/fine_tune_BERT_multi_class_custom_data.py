import torch, os

######################################

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

######################################

import pandas as pd

data = pd.read_csv("MyData/TTC4900.csv")
data = data.sample(frac=1.0, random_state=42)

print(data.head())

######################################

labels=["teknoloji","ekonomi","saglik","siyaset","kultur","spor","dunya"]

NUM_LABELS = len(labels)

id2label={i:l for i,l in enumerate(labels)}
label2id={l:i for i,l in enumerate(labels)}

print( label2id )

data["labels"] = data.category.map(lambda x: label2id[x.strip()])

print( data.head() )

####################################

print(   data.category.value_counts()  )   ##.plot(kind='pie')   ##, figsize=(8,8))

####################################

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-uncased", max_length=512)


from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased", 
                       num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)

model.to(device)

##################################


SIZE = data.shape[0]

train_texts= list(data.text[:SIZE//2])
val_texts=   list(data.text[SIZE//2:(3*SIZE)//4 ])
test_texts=  list(data.text[(3*SIZE)//4:])

train_labels= list(data.labels[:SIZE//2])
val_labels=   list(data.labels[SIZE//2:(3*SIZE)//4])
test_labels=  list(data.labels[(3*SIZE)//4:])

###################################

print(   len(train_texts), len(val_texts), len(test_texts)   )

###################################

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings   = tokenizer(val_texts,   truncation=True, padding=True)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True)

##################################

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

################################

train_dataset = MyDataset(train_encodings, train_labels)
val_dataset   = MyDataset(val_encodings,   val_labels)
test_dataset  = MyDataset(test_encodings,  test_labels)

################################

from transformers import TrainingArguments, Trainer

################################

from sklearn.metrics import accuracy_score, f1_score 

def compute_metrics(pred): 
    labels = pred.label_ids 
    preds  = pred.predictions.argmax(-1) 
    f1 = f1_score(labels, preds, average='macro') 
    acc = accuracy_score(labels, preds) 
    return {
        'Accuracy': acc,
        'F1': f1
    }


######################################

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./TTC4900Model', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    # Number of steps used for a linear warmup
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
   # TensorBoard log directory
    logging_dir='./multi-class-logs',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    fp16=True
    #load_best_model_at_end=True
)

#########################################


trainer = Trainer(
    # the pre-trained model that will be fine-tuned
    model=model,
     # training arguments that we defined above
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics= compute_metrics
)

res_train = trainer.train()

print(   res_train   )

########################################



q = [trainer.evaluate(eval_dataset=data) for data in [train_dataset, val_dataset, test_dataset]]

pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]

########################################



from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs, probs.argmax(),model.config.id2label[probs.argmax().item()]

#######################################

# Example #1

text = "Fenerbahçeli futbolcular kısa paslarla hazırlık çalışması yaptılar"
print(predict(text))

######################################

# saving the fine tuned model & tokenizer

model_path = "turkish-text-classification-model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

#####################################


model_path = "turkish-text-classification-model"

from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

######################################

r1 = nlp("Sinemada hangi filmler oynuyor bugün")
print(r1)
#[{‘label': 'kultur', 'score': 0.897723913192749}]

r2 = nlp("Dolar ve Euro bugün yurtiçi piyasalarda yükseldi")
print(r2)
#[{‘label': 'ekonomi', 'score': 0.9639127254486084}]

r3 = nlp("Bayern Münih ile Barcelona bugün karşı karşıya geliyor. Maçı İngiliz hakem James Watts yönetecek!")
print(r3)
#[{‘label': 'spor', 'score': 0.9791778922080994}]

