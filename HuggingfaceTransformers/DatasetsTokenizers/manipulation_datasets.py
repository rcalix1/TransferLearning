## >> pip install datasets

from datasets import load_dataset
import pandas as pd
import pprint

###########################################

cola = load_dataset('glue', 'cola')

cola_train = load_dataset('glue', 'cola', split='train')

cola_selection = load_dataset('glue', 'cola', split='train[:300]+validation[-30:]')

## first 50% of train and last 30%of validation
cola_selection2 = load_dataset('glue', 'cola', split='train[:50%]+validation[-30%:]')

print(   cola_selection[6, 19, 44]     )

print(   cola_selection['label'][:15]   )
cola_selection.sort('label')
print(   cola_selection['label'][:15]   )

##############################################

print(   cola_selection.shuffle(seed=42)[2:5]    )


##############################################
## dataset filter and map fucntion

## filter for sentences in cola with the word "kick"

cola_selection = load_dataset('glue', 'cola', split='train[:100%]+validation[-30%:]')

result = cola_selection.filter(lambda s: "kick" in s['sentence'])

print(     result["sentence"][:3]      )

#############################################

## get a specific class

print("sentences of a specific class")
print(   cola_selection.filter(lambda s: s['label'] == 1)["sentence"][:3]      )

print(   cola_selection.filter(lambda s: s['label'] == cola_selection.features['label'].str2int('acceptable'))["sentence"][:3]      )

#############################################

## using map to modify data or add fields

cola_new = cola_selection.map(     lambda e:{   'len': len(  e['sentence']   )    }       )
print(        pd.DataFrame(    cola_new[0:3]      )         )

############################################

cola_cut = cola_new.map(   lambda e: {'sentence': e['sentence'][:20]+ '_'}    )
print(       pd.DataFrame(    cola_cut[:3]       )     )

#############################################

## load your own dataset

from datasets import load_dataset

'''

data1 = load_dataset('csv', data_files='/data/a.csv', delimiter="\t")
data2 = load_dataset('csv', data_files=['/data/a.csv', '/data/b.csv', '/data/c.csv'], delimiter="\t")
data3 = load_dataset('csv', data_files={'train':['/data/a.csv','/data/b.csv'], 'test':['/data/c.csv']}, delimiter="\t")

## for json and txt

data_json = load_dataset('json', data_files='a.json')
data_text = load_dataset('text', data_files='a.txt')

'''

data_iris = load_dataset('csv', data_files='MyData/iris.csv', delimiter=",")
print(  data_iris   )
print(  data_iris['train'][:3] )


##################################################

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#sequence = "A Titan RTX has 24GB of VRAM"
#tokenized_sequence = tokenizer.tokenize(sequence)

cola3 = load_dataset('glue', 'cola')

encoded_data = cola3.map(     lambda e:  tokenizer(   e['sentence'], padding=True, truncation=True, max_length=12), 
                                                        batched=True, batch_size=1000       )

print(   cola_selection   )
print(   encoded_data     )

print(   encoded_data['test'][12]    )
