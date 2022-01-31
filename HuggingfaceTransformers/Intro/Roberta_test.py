
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaModel.from_pretrained('roberta-base')

text = "the cat is so sad ."

encoded_input = tokenizer(text, return_tensors='pt')

output = model(   **encoded_input    )

print(  output   )

##############################################

from transformers import pipeline

import pandas as pd

fillmask = pipeline('fill-mask', model='roberta-base', tokenizer=tokenizer)

res = pd.DataFrame(  fillmask("The cat is so <mask> .>")   )

print(  res   )

#############################################

print(    tokenizer.mask_token     )

#############################################
