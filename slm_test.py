from slm import *
from glob import glob

context_length = 2

def get_model(context_length):
    #tokenizer = WhitespaceTokenizer()
    #tokenizer = CharacterTokenizer()
    #tokenizer = SpaceTokenizer()
    tokenizer = Gpt2Tokenizer()

    embedder = NullEmbedder()

    predictor = FrequencyTablePredictor(context_length)

    model = LanguageModel(
        tokenizer=tokenizer,

        #embedder=embedder,

        predictor=predictor
    )

    return model

# Usage starts here


# Training
def get_initial_model(filename):
    model = get_model(context_length)
    text = open(filename, encoding='utf-8').read()
    text = text.replace("\n"," ")
    tokens = model.tokenizer(text)

    model.train(tokens)
    return model

model1 = get_initial_model('data/1200_en.txt')
model2 = get_initial_model('data/1200_es.txt')

gen_n_tokens = 100

newline_token = model1.tokenizer('\n')[0]
wspace_token = model1.tokenizer(' ')[0]
excl_token = model1.tokenizer('!')[0]
question_token = model1.tokenizer('?')[0]
period_token = model1.tokenizer('.')[0]

stop_list = (excl_token, question_token, period_token)

def get_blurb(model):
    #init_keys = [k for k in model.predictor.follower_table.keys() if k[0] == newline_token]
    init_contexts = list(model.predictor.follower_table.keys())

    proposals = []
    for context in init_contexts:
        context_proposal = model.tokenizer.decode(context)[0]
        #print(context_proposal)
        if context_proposal.isupper(): 
            proposals.append(context)
        if len(proposals) == 0: initial_context = random.choice(init_contexts)
        else: initial_context = random.choice(proposals)
        
    #initial_context = random.choice(init_keys)

    out_tokens = []
    for i, token in enumerate(model.generate(initial_context)):
        
        out_tokens.append(token)
        if i > 100 and token in stop_list:
            break
    return out_tokens

def go_chat(model1, model2, iters):
    for i in range(iters):
        model1_out = get_blurb(model1)
        model2.train(model1_out)

        print("EN:", model1.tokenizer.decode(model1_out).strip())
        print()

        model2_out = get_blurb(model2)
        model1.train(model2_out)
        
        print("ES:", model2.tokenizer.decode(model2_out).strip())
        print()
go_chat(model1,model2,100)
#print(model1.predictor.follower_table)

#print()
#model = get_model(context_length)
#text = open('sample_data/subs_1000.en', encoding='utf-8').read()
#tokens = model.tokenizer(text)
#print(tokens)
    #generated_tokens = model.generate(initial_context, 100)
    #generated_tokens = list(generated_tokens)

    #generated_text = model.tokenizer.decode(generated_tokens)