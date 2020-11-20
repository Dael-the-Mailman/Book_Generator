import torch
import torch.optim as optim
import io
import os

from model import Transformer
from utils.custom_utils import load_vocab, create_sentence
from utils.utils import load_checkpoint

def write(string, filename):
    with io.open(os.path.join('./output/', filename), "a") as output:
        output.write('\n{}'.format(string))

input_vocab, output_vocab = load_vocab()

print("Input Vocab Size: {}".format(len(input_vocab.vocab)))
print("Output Vocab Size: {}".format(len(output_vocab.vocab)))

# Model hyperparameters (Must be the same as training)
src_vocab_size = len(input_vocab.vocab)
trg_vocab_size = len(output_vocab.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.5
max_len = 100
forward_expansion = 4
src_pad_idx = input_vocab.vocab.stoi["<pad>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0003)

load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

seed = input("Starting sentence: ")
filename = "Custom Output.txt"

curr_len = len(seed.split(' '))
total_words = curr_len

write(seed, filename)

first_time = True
prev_pred = ''

while total_words < 90000:
    input_sentence = seed if first_time else prev_pred
    prediction = create_sentence(
            model, input_sentence, input_vocab, output_vocab, device, max_length=50
        )[1:-1]
    curr_len = len(prediction)
    prediction = ' '.join(prediction)
    write(prediction, filename)
    prev_pred = prediction
    total_words += curr_len

    print("Prediction:", prediction)
    print("Prediction String Length:", curr_len)
    print("Total Words:", total_words)
    print("\n")
    if first_time:
        first_time = False