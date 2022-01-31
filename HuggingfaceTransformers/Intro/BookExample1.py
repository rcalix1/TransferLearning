from transformers import BertTokenizer
import pandas as pd

##########################################


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Using transformer is easy!"

print(    tokenizer(text)    )

encoded_input = tokenizer(text, return_tensors="pt")   ## pt is for pytorch tensors

print(  encoded_input   )

#####################################################

from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")

output = model(**encoded_input)

print(output)

####################################################

from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')

result = unmasker("the man worked as a [MASK].")     

print(result)

print(   pd.DataFrame(result)     )




