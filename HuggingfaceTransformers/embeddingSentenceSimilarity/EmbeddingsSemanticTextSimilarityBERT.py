## Sentence Embeddings 
## and semantic text similarity

## >>pip install torch
## >>pip install transformers
## >>pip install protobuf
## >>pip install sentence-transformers
## >>pip install dataset
## >>pip install flair

##############################################

import pandas as pd
import torch
import torch
import numpy as np
from flair.data import Sentence


##############################################

similar=[
("A black dog walking beside a pool.","A black dog is walking along the side of a pool."),
("A blonde woman looks for medical supplies for work in a suitcase.	"," The blond woman is searching for medical supplies in a suitcase."),
("A doubly decker red bus driving down the road.","A red double decker bus driving down a street."),
("There is a black dog jumping into a swimming pool.","A black dog is leaping into a swimming pool."),
("The man used a sword to slice a plastic bottle.	","A man sliced a plastic bottle with a sword.")
]


print(   pd.DataFrame(similar, columns=["sen1", "sen2"])    )

################################################

dissimilar= [
("A little girl and boy are reading books. ", "An older child is playing with a doll while gazing out the window."),
("Two horses standing in a field with trees in the background.", "A black and white bird on a body of water with grass in the background."),
("Two people are walking by the ocean." , "Two men in fleeces and hats looking at the camera."),
("A cat is pouncing on a trampoline.","A man is slicing a tomato."),
("A woman is riding on a horse.","A man is turning over tables in anger.")
]

print(    pd.DataFrame(dissimilar, columns=["sen1", "sen2"])    )

#################################################

def sim(s1,s2):
  # cosine similarity function outputs in the range 0-1
  s1=s1.embedding.unsqueeze(0)
  s2=s2.embedding.unsqueeze(0)
  sim = torch.cosine_similarity(s1,s2).item() 
  return np.round(sim,2)
  
##################################################

def evaluate(embeddings, myPairList):
  # it evaluates embeddings for a given list of sentence pair
  scores=[]
  for s1, s2 in myPairList:
    s1,s2=Sentence(s1), Sentence(s2)
    embeddings.embed(s1)
    embeddings.embed(s2)
    score=sim(s1,s2)
    scores.append(score)
  return scores, np.round(np.mean(scores),2)

###################################################
## Average word embeddings with GloVe

from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

glove_embedding = WordEmbeddings('glove')
glove_pool_embeddings = DocumentPoolEmbeddings([glove_embedding])

print("--------------------------------")
print("Glove just average embeddings")
print(    evaluate(glove_pool_embeddings, similar)     )
print(    evaluate(glove_pool_embeddings, dissimilar)  )

####################################################
## Considers sequence
## RNN based GRU


from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings
gru_embeddings = DocumentRNNEmbeddings([glove_embedding])

print("--------------------------------")
print("GRU RNN embeddings") 

print(    evaluate(gru_embeddings, similar)      )
print(    evaluate(gru_embeddings, dissimilar)   )

#####################################################
## The following execution instantiates a "bert-base-uncased" model that 
## pools the final layer

## A non-specific BERT

from flair.embeddings import TransformerDocumentEmbeddings


bert_embeddings = TransformerDocumentEmbeddings('bert-base-uncased')

print("--------------------------------")
print("BERT non-specialized embeddings")
print("notice it is not better than GloVe")
print("actually a bit worse")

print(   evaluate(bert_embeddings, similar)       )
print(   evaluate(bert_embeddings, dissimilar)    )


#####################################################
## Sentence BERT
## Now a specialized BERT for this task

from flair.embeddings import SentenceTransformerDocumentEmbeddings

sbert_embeddings = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')

print("--------------------------------")
print(  "Notice, this one actually does what we want"  )
print(   evaluate(sbert_embeddings, similar)   )
print(   evaluate(sbert_embeddings, dissimilar)   )

#####################################################
## now we will do a harder similarity test
## tricky sentences
## contradicting sentences

tricky_pairs=[
("An elephant is bigger than a lion", "A lion is bigger than an elephant"),
("the cat sat on the mat", "the mat sat on the cat")
]

print("--------------------------------")
print("GRU here does better, because sequence matters in RNNs?")

print(    evaluate(glove_pool_embeddings, tricky_pairs)    )

print(    evaluate(gru_embeddings, tricky_pairs)           )

print(    evaluate(bert_embeddings, tricky_pairs)          )

print(    evaluate(sbert_embeddings, tricky_pairs)         )

#####################################################
## expected outputs

'''
--------------------------------
Glove just average embeddings
([0.97, 0.99, 0.97, 0.99, 0.98], 0.98)
([0.94, 0.97, 0.94, 0.92, 0.93], 0.94)
--------------------------------
GRU RNN embeddings
([0.99, 1.0, 0.95, 1.0, 0.92], 0.97)
([0.88, 1.0, 0.93, 0.81, 0.91], 0.91)
--------------------------------
BERT non-specialized embeddings
notice it is not better than GloVe
actually a bit worse
([0.85, 0.9, 0.96, 0.91, 0.89], 0.9)
([0.93, 0.94, 0.86, 0.93, 0.92], 0.92)
--------------------------------
Notice, this one actually does what we want (SBERT)
([0.98, 0.95, 0.96, 0.99, 0.98], 0.97)
([0.48, 0.41, 0.19, -0.05, 0.0], 0.21)
--------------------------------
GRU here does better, because sequence matters in RNNs?
([1.0, 1.0], 1.0)
([0.84, 0.75], 0.8)
([1.0, 0.98], 0.99)
([0.93, 0.97], 0.95)

'''

#####################################################
## We need a BERT specialized for this problem
## of condradicting sentences
## there is a model from XNLI for this task
## there is a model tp detect the semantics of 2 sentence
## pairs with 3 classes: neutral, contradiction, entailment
## we use a fine tune XLM-Roberta model trained on XNLI

from transformers import AutoModelForSequenceClassification, AutoTokenizer

nli_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-xnli')
tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli')

for premise, hypothesis in tricky_pairs:
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation_strategy='only_first')
    
    logits = nli_model(x)[0]
    print(f"Premise: {premise}")
    print(f"Hypothesis: {hypothesis}")
    print("Top Class")
    ids = np.argmax(   logits[0].detach().numpy()   )
    print(   nli_model.config.id2label[ids]   )
    print("full softmax scores: ")
    for i in range(3):
        print(nli_model.config.id2label[i],        
              logits.softmax(dim=1)[0][i].detach().numpy()
        )
    print("="*20)



#####################################################

print("<<<<<<DONE>>>>>>>")





