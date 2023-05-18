# word2id_{}.json, word_embedding_{}.npy 생성
import os
import numpy as np
import json
from collections import defaultdict
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec

from load_data import load_corpus, get_dir_and_base_name

DATA_NAME = 'CD_5'
# DATA_NAME = 'Clothing_5'
#DATA_NAME = 'Digital_Music_5'

ROOT =  '/home/yeonghwa/workspace/reviewgraph/final_cd/'
DATA_PATH = ROOT + 'dataset/{}/{}.json'.format(DATA_NAME, DATA_NAME)

GLOVE_PATH = ROOT + 'dataset/glove.6B.100d.txt'
GLOVE_TMP_PATH = ROOT + 'dataset/glove.6B.100d.word2vec.txt'

""" DATASET LIST
'Office_Products_5'
'Instant_Video_5'
'Digital_Music_5'
'Sports_and_Outdoors_5'
'Clothing_5'
'Toys_and_Games_5'
'Health_and_Personal_Care_5'
'CDs_and_Vinyl_5'
'Movies_and_TV_5'
'Baby_5'
'Yelp2013'
---
'Cell_Phones_and_Accessories_5'
'Beauty_5'
'Apps_for_Android_5'
"""

class Args:
    dataset_name = DATA_NAME
    dataset_path = DATA_PATH

    vocab_size = 50000
    embedding_dim = 100

    args_str = 'embed_dim_{}'.format(embedding_dim)


dir_path, _ = get_dir_and_base_name(Args.dataset_path)

Args.word2id_path = \
    '{}/word2id_{}.json'.format(dir_path, Args.args_str)
Args.embedding_path = \
    '{}/word_embedding_{}.npy'.format(dir_path, Args.args_str)


def load_glove():
    tmp_file = get_tmpfile(GLOVE_TMP_PATH)

    if not os.path.exists(GLOVE_TMP_PATH):
        glove_file = datapath(GLOVE_PATH)

        # _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

    word2id = model.key_to_index
    embed = model.vectors

    return word2id, embed


if __name__ == '__main__':
    args = Args
    print('Load Word2Vec from Glove, on dataset {}'.format(args.dataset_name))

    sentences = load_corpus(args.dataset_path)
    sentences = [x.split() for x in sentences]

    corpus_word_num = 0
    corpus_word_counter = defaultdict(int)
    for words in sentences:
        corpus_word_num += len(words)
        for word in words:
            corpus_word_counter[word] += 1

    corpus_word_counter = sorted(corpus_word_counter.items(),
                                 key=lambda x: x[1], reverse=True)
    corpus_word_counter = corpus_word_counter[:args.vocab_size]
    corpus_word_counter = dict(corpus_word_counter)

    glove_word2id, glove_embed = load_glove()
    corpus_word_set = set(corpus_word_counter.keys())
    glove_word_set = set(glove_word2id.keys())

    used_words = list(corpus_word_set & glove_word_set)
    unused_words = list(corpus_word_set - glove_word_set)

    unused_words_counter = {k: corpus_word_counter[k] for k in unused_words}
    unused_words_counter = dict(sorted(unused_words_counter.items(),
                                       key=lambda x: x[1], reverse=True))

    ds_word2id = {'<PAD>': 0}
    ds_embed = list()
    ds_embed.append(np.zeros(glove_embed.shape[1]))

    # used words
    for w in used_words:
        ds_word2id[w] = len(ds_word2id)
        ds_embed.append(glove_embed[glove_word2id[w]])

    for w in unused_words:
        ds_word2id[w] = len(ds_word2id)
        ds_embed.append(np.random.normal(.0, 1., glove_embed.shape[1]))

    ds_embed = np.stack(ds_embed, axis=0)

    with open(args.word2id_path, 'w') as f:
        json.dump(ds_word2id, f)
        
    np.save(args.embedding_path, ds_embed)
