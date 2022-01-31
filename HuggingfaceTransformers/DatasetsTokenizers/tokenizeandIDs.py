## These tokens can then be converted into IDs which are understandable by the model


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"

tokenized_sequence = tokenizer.tokenize(sequence)

print(tokenized_sequence)

## ['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']

inputs = tokenizer(sequence)

encoded_sequence = inputs["input_ids"]
print(encoded_sequence)


## [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]

decoded_sequence = tokenizer.decode(encoded_sequence)


print(decoded_sequence)

## [CLS] A Titan RTX has 24GB of VRAM [SEP]
