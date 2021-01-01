# Book Generator
[![I created an AI to write books](http://img.youtube.com/vi/8V6f5BLcTnI/0.jpg)](http://www.youtube.com/watch?v=8V6f5BLcTnI)

---

## Description

Creates a text file that is 90,000 words long from a single starting sentence. Was trained on the NLTK gutenberg dataset. The model used was a sequence to sequence transformer model based off of [Aladdin Persson's](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/seq2seq_transformer) seq2seq transformer model. I implemented k-fold cross validation during the training.

#### Technologies

- Python
- PyTorch
- Numpy
- Pandas
- NLTK

---

## Installation
```
cd ./file/location
git clone https://github.com/Dael-the-Mailman/Book_Generator.git
conda create --name bookgenerator
conda activate bookgenerator
pip install numpy pandas nltk 

# Check which version of pytorch is appropriate for your system(e.g. CPU Only, CUDA version, etc.)
pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
---
## How To Use
```
conda activate bookgenerator

# Use GPT2 model
python text_generator.py

# Use Custom model
python custom_generator.py
```
