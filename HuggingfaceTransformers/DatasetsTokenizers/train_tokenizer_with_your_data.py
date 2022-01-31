import pandas as pd
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast


#######################################

imdb_df = pd.read_csv("MyData/IMDB_dataset.csv")
reviews = imdb_df.review.to_string(index=None)
with open("corpus.txt", "w") as f:
    f.writelines(reviews)

#######################################

bert_wordpiece_tokenizer = BertWordPieceTokenizer()
bert_wordpiece_tokenizer.train("corpus.txt")

print(    bert_wordpiece_tokenizer.get_vocab()    )

bert_wordpiece_tokenizer.save_model("tokenizer")

tokenizer2 = BertWordPieceTokenizer.from_file("tokenizer/vocab.txt")


########################################

tokenized_sentence = tokenizer2.encode("oh it works just fine")

print(   tokenized_sentence          )
print(   tokenized_sentence.tokens   )



