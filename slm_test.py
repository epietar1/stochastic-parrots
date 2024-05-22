from slm import *

text = open("sample_data/blowin_in_the_wind_verses.txt")
text = text.read()


context_length = 5

#tokenizer = WhitespaceTokenizer()
tokenizer = CharacterTokenizer()

tokens = tokenizer(text)

embedder = NullEmbedder()

predictor = FrequencyTablePredictor(context_length)

model = LanguageModel(
    tokenizer=tokenizer,
    embedder=embedder,
    predictor=predictor
)

model.train(tokens)
initial_context = tokens[:2]

generated_tokens = model.generate(initial_context, 100)
generated_text = model.tokenizer.decode(generated_tokens)

print(generated_text)