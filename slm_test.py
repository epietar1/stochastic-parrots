from slm import *
from glob import glob
from parrot_utils import *
import matplotlib.pyplot as plt

# Usage starts here

context_length = 3
model1_fname = 'data/20000_cr.txt'
model2_fname = 'data/20000_en.txt'
modelref_fname = 'data/20000_encr.txt'

model1_text = open(model1_fname, encoding='utf-8').read().replace("\n"," ")
model2_text = open(model2_fname, encoding='utf-8').read().replace("\n"," ")

model1 = get_initial_model(model1_fname, context_length)
model2 = get_initial_model(model2_fname, context_length)
model_ref = get_initial_model(modelref_fname, context_length)

model_ref_entropy = get_gram_entropy(model_ref.predictor.follower_table)[1][0]

print("Initial likelihoods:")
#print(f"Model 1 log likelihood for own text {calculate_log_probability(model1,model1_text)}")
#print(f"Model 2 log likelihood for own text {calculate_log_probability(model2,model2_text)}")
print()
#print(f"Model 1 log likelihood for model 2 text {calculate_log_probability(model1,model2_text)}")
#print(f"Model 2 log likelihood for model 1 text {calculate_log_probability(model2,model1_text)}")

"""
gen_n_tokens = 100

newline_token = model1.tokenizer('\n')[0]
wspace_token = model1.tokenizer(' ')[0]
excl_token = model1.tokenizer('!')[0]
question_token = model1.tokenizer('?')[0]
period_token = model1.tokenizer('.')[0]

stop_list = (excl_token, question_token, period_token)
"""

m_count_init = 0
model1_count_init = len(model1.predictor.follower_table.keys())
model2_count_init = len(model2.predictor.follower_table.keys())

for key in model2.predictor.follower_table.keys():

    if key in model1.predictor.follower_table.keys():
        m_count_init += 1


print("Initial counts for contexts")
print(f"mutual: {m_count_init}")
print(f"total model1: {model1_count_init}")
print(f"total model2: {model2_count_init}")

ent1, ent2 = train(model1,model2,1500,print_every = 100, init1 = ('J','a',' '), init2 = ('O','u','i'), print_out=False)

plt.plot(ent1,label = "Model 1")
plt.plot(ent2, label = "Model 2")
plt.axhline(y=model_ref_entropy, color='r', linestyle='-', label = "Combined")
plt.title("Entropy of language models")
plt.ylabel("Entropy")
plt.xlabel("Iteration (/ report every)")
plt.legend()
plt.show()
#test(model1, model2, 10)


print("Final likelihoods (for original text):")
#print(f"Model 1 log likelihood for own text {calculate_log_probability(model1,model1_text)}")
#print(f"Model 2 log likelihood for own text {calculate_log_probability(model2,model2_text)}")
print()
#print(f"Model 1 log likelihood for model 2 text {calculate_log_probability(model1,model2_text)}")
#print(f"Model 2 log likelihood for model 1 text {calculate_log_probability(model2,model1_text)}")
print()