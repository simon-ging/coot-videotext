import json
import re
from pathlib import Path
from typing import List

import nltk
import numpy as np

import utils


def preprocess_bert_paragraph(
        paragraph: List[str]) -> List[List[str]]:
    new_paragraph = []
    for i, sentence in enumerate(paragraph):
        new_sentence = []
        if i == 0:
            new_sentence.append("[CLS]")
        preproc_sentence = preprocess_bert_sentence(sentence)
        for word in preproc_sentence:
            new_sentence.append(word.strip())
        new_paragraph.append(new_sentence)
    return new_paragraph


def preprocess_bert_sentence(sentence_str: str) -> List[str]:
    if sentence_str[-1] == ".":
        sentence_str = sentence_str[:-1]
    sentence_str = sentence_str.replace(". ", " [SEP] ")
    sentence_str += " [SEP] "
    sentence_str = re.sub(r"\s+", " ", sentence_str).strip()
    words = sentence_str.split(" ")
    return words


class Vocab(object):
    """Simple vocabulary wrapper. Needed to load pickle file"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def load(self, file):
        self.word2idx, self.idx2word, self.idx = json.load(Path(file).open(
            "rt", encoding="utf8"))

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['UNK']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


# noinspection PyUnresolvedReferences
class GloveLookup(object):
    def __init__(self):
        glove_path = Path("glove_vocab")

        # vocab_path = glove_path / "activitynet_vocab.dill"
        # self.vocab = dill.load(open(vocab_path, 'rb'))
        # vocab = Vocab()
        # vocab.word2idx = self.vocab.word2idx
        # vocab.idx2word = self.vocab.idx2word
        # vocab.idx = self.vocab.idx
        # vocab.dump(glove_path / "vocab.json")
        # exit()

        self.vocab = Vocab()
        self.vocab.load(glove_path / "vocab.json")
        mapping_path = glove_path / "precomp_anet_w2v_total.npz"
        npz_file = np.load(str(mapping_path))
        np_arr = npz_file[npz_file.files[0]]
        np_arr = np_arr.astype(np.float)
        self.shared_array = utils.make_shared_array(np_arr)
        assert np_arr.shape[0] == len(self.vocab)
        self.feature_dim = 300

    def __getitem__(self, word):
        word_idx = self.vocab(word)
        vector = self.shared_array[word_idx]
        return vector, True


def preprocess_glove_paragraph(paragraph: List[str]) -> List[List[str]]:
    list_of_list_of_words = []
    for sentence in paragraph:
        while True:
            sentence = sentence.strip()
            if sentence.endswith("."):
                sentence = sentence[:-1]
            else:
                break
        sentence = sentence.replace(". ", " addeostokenhere ")
        try:
            list_of_words = nltk.tokenize.word_tokenize(sentence.lower())
        except LookupError:
            print("nltk is missing some resource")
            print("running nltk.download('punkt')")
            nltk.download('punkt')
            list_of_words = nltk.tokenize.word_tokenize(sentence.lower())
        new_list_of_words = []
        for word in list_of_words:
            if len(word) > 2 and word[0] == "'":
                new_list_of_words += ["'", word[1:]]
            else:
                new_list_of_words.append(word)
        list_of_words = new_list_of_words
        fix_indices = []
        for i, word in enumerate(list_of_words):
            if word == "addeostokenhere":
                fix_indices.append(i)
        for i in fix_indices:
            list_of_words.pop(i)
            list_of_words.insert(i, '.')
        list_of_words += ["."]
        list_of_list_of_words.append(list_of_words)
    return list_of_list_of_words
