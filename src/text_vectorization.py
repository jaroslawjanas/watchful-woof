import torch
import torch.nn as nn
from collections import defaultdict


class TextVectorization(nn.Module):
    def __init__(self, max_vocabulary, max_tokens):
        super(TextVectorization, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_tokens = max_tokens
        self.max_vocabulary = max_vocabulary
        self.word_dictionary = dict()
        self.dictionary_size = 0

    def adapt(self, dataset):
        # Calculate word frequencies
        word_frequencies = defaultdict(int)

        for text in dataset:
            for word in text[0].split():
                word_frequencies[word] += 1

        # Sort the dictionary by word frequencies in descending order
        sorted_word_frequencies = dict(
            sorted(
                word_frequencies.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        # Take the top (max_vocabulary - 2) most frequent words
        # since indices 0 and 1 are reserved for padding and missing words respectively
        most_frequent = list(sorted_word_frequencies.items())[:self.max_vocabulary - 2]
        # Note that len(most_frequent) does not necessarily equal
        # (max_vocabulary - 2), since there could be less words overall
        # than the max_vocabulary limit
        self.dictionary_size = len(most_frequent) + 2

        # Note starting at 2 since 0 (padding) and 1 (missing) are reserved
        for word_value, (word, frequency) in enumerate(most_frequent, 2):
            self.word_dictionary[word] = word_value

        # if len(self.word_dictionary) < self.max_vocabulary:
        #     raise ValueError(
        #         f"Current size of the dictionary ({len(self.word_dictionary)}) "
        #         f"exceeds the defined limit ({self.max_vocabulary})"
        #     )

    def vocabulary_size(self):
        return self.dictionary_size

    def dictionary(self):
        return self.word_dictionary

    def forward(self, batch_x):
        try:
            batch_text_vectors = torch.zeros((len(batch_x), self.max_tokens), dtype=torch.int32, device=self.device)

            for i, text in enumerate(batch_x):

                # Split the text and tokenize it
                words = text.split()[:self.max_tokens]

                for pos, word in enumerate(words):
                    batch_text_vectors[i, pos] = self.word_dictionary.get(word, 1)

            return batch_text_vectors

        except IndexError:
            print("Looks like you are out of indices")
