corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

unique_chars = set()

for doc in corpus:
    for char in doc:
        unique_chars.add(char)

vocab = list(unique_chars)
vocab.sort() # For consistent ordering of characters

end_of_word = '</w>'
vocab.append(end_of_word)

print("Initial Vocabulary:")
print(vocab)
print(f"Vocabulary Size: {len(vocab)}") 

word_splits = {}
for doc in corpus:
    words = doc.split(' ')
    for word in words:
       if word:
           char_list = list(word) + [end_of_word]

           word_tuple = tuple(char_list)
           if word_tuple not in word_splits:
               word_splits[word_tuple] = 0
           word_splits[word_tuple] += 1

import collections

def get_pair_stats(splits):
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_counts[pair] += freq
    return pair_counts



