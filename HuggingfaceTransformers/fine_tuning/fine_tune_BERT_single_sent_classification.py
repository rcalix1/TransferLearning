import torch


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

model_path = 'distilbert-base-uncased'

tokenizer = DistilBertTokenizerFast.from_pretrained( model_path  )

model = DistilBertForSequenceClassification.from_pretrained(model_path, id2label={0: "NEG", 1: "POS"},
                     label2id={"NEG": 0, "POS": 1})


#####################################################

from datasets import load_dataset

imdb_train = load_dataset('imdb', split="train")

imdb_test = load_dataset('imdb', split="test[:6250]+test[-6250:]")

imdb_val = load_dataset('imdb', split='test[6250:12500]+test[-12500:-6250]')

print(  imdb_train.shape  )
print(  imdb_test.shape   )
print(  imdb_val.shape    )

####################################################

enc_train = imdb_train.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True,
                  batch_size=1000  )

enc_test = imdb_test.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, 
                  batch_size=1000)

enc_val = imdb_val.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, 
                  batch_size=1000)

#####################################################


import pandas as pd

print(  pd.DataFrame(enc_train)   )

#####################################################

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./MyIMDBModel', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    logging_strategy='steps', 
    # TensorBoard log directory               
    logging_dir='./logs',            
    logging_steps=50,
    # other options : no, steps
    evaluation_strategy="steps",
    fp16 = cuda.is_available(),
    save_strategy="epoch"
    #load_best_model_at_end=True
)

##################################################

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'Accuracy': acc,
        'f1': f1
    }

################################################

trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,
    # training and validation dataset                 
    train_dataset=enc_train,         
    eval_dataset=enc_val,            
    compute_metrics= compute_metrics
)

results = trainer.train()



###############################################

# saving the best fine-tuned model & tokenizer
model_save_path = "MyBestIMDBModel"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

#################################################

def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=250, return_tensors="pt").to(device)
    outputs = model(inputs["input_ids"].to(device),inputs["attention_mask"].to(device))
    probs = outputs[0].softmax(1)
    return probs, probs.argmax()


model.to(device)
text = "I didn't like the movie since it bored me "
res = get_prediction(text)[1].item()
print(res)

#############################################

from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast
model = DistilBertForSequenceClassification.from_pretrained("MyBestIMDBModel")
tokenizer= DistilBertTokenizerFast.from_pretrained("MyBestIMDBModel")
nlp= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


r1 = nlp("the movie was very impressive")
r2 = nlp("the script of the picture was very poor")

print(r1)
print(r2)



