## These require GPU

* Examples from HF
* RESNET_one_GPU.ipynb -> Scholar GPU cloud with 1 GPU and Python 3.8
* https://huggingface.co/docs/datasets/en/nlp_load


## code fix

```


training_args = TrainingArguments(
    output_dir=f"{model_checkpoint}-wikitext2",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs"
)

```
