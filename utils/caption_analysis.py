import os
import nltk
from nltk import word_tokenize, pos_tag
import torch
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt

data_root = '/mnt/jaewoo4tb/textraj/preprocessed_1st/jrdb_v1/'

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# analysis perprocessed datas
files = glob.glob(os.path.join(data_root, '*.pt'))
files.sort()

words = []
filters = ['VBG']

idx = 0
for file in tqdm(files):
    words.append({})
    data = torch.load(file)
    for frame, frame_data in data.items():
        for person, person_data in frame_data.items():
            if ('description' not in person_data) or (person_data['description'] is None):
                continue

            caption = person_data['description'].lower()
            sentence = word_tokenize(caption)
            pos_tags = pos_tag(sentence)
            for word, tag in pos_tags:
                if tag in filters:
                    if word not in words[idx]:
                        words[idx][word] = 1
                    else:
                        words[idx][word] += 1

    idx += 1

# concat train/val
val_idxs = [0, 13, 14, 18]

train_words = {}
val_words = {}

for idx, word in enumerate(words):
    for w, c in word.items():
        if idx in val_idxs:
            if w not in val_words:
                val_words[w] = c
            else:
                val_words[w] += c
        else:
            if w not in train_words:
                train_words[w] = c
            else:
                train_words[w] += c


# plot histogram
def plot_word_histogram(word_data, ax, split, max_bars=10):
    sorted_word_data = dict(sorted(word_data.items(), key=lambda item: item[1], reverse=True))
    
    filtered_word_data = dict(list(sorted_word_data.items())[:max_bars])
    
    words = list(filtered_word_data.keys())
    frequencies = list(filtered_word_data.values())
    
    ax.bar(words, frequencies, color='skyblue')
    ax.set_xlabel('Word')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{split} Word-Frequency Histogram')
    ax.tick_params(axis='x', rotation=45)  

fig, axs = plt.subplots(1, 2, figsize=(10, 6))

plot_word_histogram(train_words, axs[0], "train", 15)
plot_word_histogram(val_words, axs[1], "val", 15)

plt.tight_layout()
plt.show()
plt.savefig("jrdb.jpg", dpi=1000)

breakpoint()