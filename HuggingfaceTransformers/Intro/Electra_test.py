
from transformers import ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')


##############################################

from transformers import pipeline

import pandas as pd

fillmask = pipeline('fill-mask', model='google/electra-small-generator', tokenizer=tokenizer)

#res = pd.DataFrame(  fillmask("The cat is very [MASK] .")   )

res = pd.DataFrame(  fillmask("The goat on the mountain will [MASK] .")  )

print(  res   )

#############################################

print(    tokenizer.mask_token     )

#############################################

