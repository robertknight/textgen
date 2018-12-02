import json

import numpy as np
import torch


class Vocabulary:
    def __init__(self, terms):
        self.terms = terms
        self.term_dict = {}
        for i, term in enumerate(terms):
            self.term_dict[term] = i
        self.eye = torch.eye(len(terms))

    def encode(self, terms):
        term_idxs = self.encode_as_indexes(terms)
        return self.eye[term_idxs]

    def encode_as_indexes(self, terms):
        return [self.term_dict[t] for t in terms]

    def decode(self, encoded):
        decoded = ""
        for i in range(len(encoded)):
            char_idx = np.argmax(encoded[i])
            decoded += self.terms[char_idx]
        return decoded

    def decode_indexes(self, encoded):
        decoded = ""
        for i in range(len(encoded)):
            decoded += self.terms[encoded[i]]
        return decoded

    def save(self, file_):
        json.dump(self.terms, file_)

    @property
    def size(self):
        return len(self.terms)

    @classmethod
    def load(cls, file_):
        terms = json.load(file_)
        return cls(terms)


def print_table(headers, items, **kwargs):
    print(",".join(headers), **kwargs)
    for idx, row in enumerate(items):
        print(f"{idx},{row}")
