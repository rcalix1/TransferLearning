## >> pip install SentencePiece

from transformers import AlbertTokenizer, AlbertModel

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

model = AlbertModel.from_pretrained("albert-base-v2")

text = "the cat is so sad ."

encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)

print(output)

######################################

import pandas as pd

from transformers import pipeline

fillmask = pipeline('fill-mask', model='albert-base-v2')
res = pd.DataFrame(fillmask("The cat is so [MASK] ."))

res1 = pd.DataFrame(fillmask("El chapo is a  [MASK] person."))
print(  res1   )

print(  res   )

res2 = pd.DataFrame(fillmask("Miguel is a  [MASK] person."))
print(  res2   )
res3 = pd.DataFrame(fillmask("Michael is a  [MASK] person."))
print(  res3   )
