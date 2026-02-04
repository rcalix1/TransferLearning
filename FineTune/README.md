# Colab

* link

## Transfer Learning

* https://github.com/huggingface/transformers/tree/main/examples/pytorch
* 

## WanDB

* import wandb
* wandb.init(mode="disabled")

## Other

* !python --version
* 


```

import numpy as np
import evaluate
from transformers import DataCollatorWithPadding

# Load the metric needed for compute_metrics
metric = evaluate.load("accuracy")

# Use a DataCollator to handle padding, which avoids the 'tokenizer' keyword issue in Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

# The Trainer initialization using data_collator instead of tokenizer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_train_dataset,
    ## eval_dataset=encoded_dataset[validation_key],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

```
