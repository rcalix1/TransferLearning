from transformers import AutoModel, pipeline, BartTokenizer, BartForConditionalGeneration, BartConfig


summarizer = pipeline("summarization")

model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')

tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

nlp = pipeline("summarization", model=model, tokenizer=tokenizer)


text = '''
Aviation is the activities surrounding mechanical flight and the aircraft industry. Aircraft includes fixed-wing and rotary-wing types, 
morphable wings, wing-less lifting bodies, as well as lighter-than-air craft such as hot air balloons and airships.
Aviation began in the 18th century with the development of the hot air balloon, an apparatus capable of atmospheric displacement through buoyancy. 
Some of the most significant advancements in aviation technology came with the controlled gliding flying of Otto Lilienthal in 1896; then a large 
step in significance came with the construction of the first powered airplane by the Wright brothers in the early 1900s. Since that time, aviation 
has been technologically revolutionized by the introduction of the jet which permitted a major form of transport throughout the world.
'''

print(text)
print('**********************')

q = nlp(text)

print(q)






