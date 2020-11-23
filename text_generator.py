import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils.custom_utils import tokenize_text
import os
import io

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

def predict(payload):
    payload_size = len(tokenizer(payload)["input_ids"])
    tokens = tokenizer.encode(payload, return_tensors="pt").to("cuda")
    max_length = 30 if payload_size < 30 else 50
    prediction = model.generate(tokens, max_length=max_length, do_sample=True)
    output = tokenizer.decode(prediction[0])
    
    return output

def write(string, filename):
    with io.open(os.path.join('./output/', filename), "a", encoding="utf-8") as output:
        output.write('\n{}'.format(string))

    # with open(os.path.join('./output/', filename),"a") as output:
    #     output.write('\n{}'.format(string))

seed = input("Starting sentence: ")
filename = "GPT2 Output.txt"

curr_len = len(seed.split(' '))
total_words = curr_len

write(seed, filename)

first_time = True

prev_pred = ''

while total_words < 90000:
    torch.cuda.empty_cache()
    prediction = predict(seed if first_time else prev_pred)
    split_pred = prediction.split(' ')[curr_len:]
    curr_len = len(split_pred)
    while curr_len == 0:
        torch.cuda.empty_cache()
        prediction = predict(seed if first_time else prev_pred)
        split_pred = prediction.split(' ')[curr_len:]
        curr_len = len(split_pred)
    prediction = ' '.join(split_pred)
    write(prediction, filename)
    prev_pred = prediction
    total_words += curr_len

    print("Prediction:", split_pred)
    print("Prediction String Length:", curr_len)
    print("Total Words:", total_words)
    print("\n")
    if first_time:
        first_time = False