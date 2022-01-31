from transformers import pipeline


print(     pipeline('sentiment-analysis')('we love you')          )
