from transformers import pipeline

'''
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence_to_classify = "I am going to france"

candidate_labels = ['travel', 'cooking', 'dancing']

result = classifier(sequence_to_classify, candidate_labels)

print(result)

'''

sequence = "I am going to france" 
label = ['travel', 'cooking', 'dancing']

from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

premise = sequence
hypothesis = f'This example is {label}.'

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')

logits = nli_model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 

entail_contradiction_logits = logits[:,[0,2]]

probs = entail_contradiction_logits.softmax(dim=1)

prob_label_is_true = probs[:,1]

