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

def merge_pair(pair_to_merge, splits):
    new_splits = {}
    (first, second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[tuple(new_symbols)] = freq
    return new_splits
            

num_merges = 15
merges = {}
current_splits = word_splits.copy()

print("Starting BPE Merges")
print(f"Initial Splits: {current_splits}")
print("-" * 30)

for i in range(num_merges):
    print(f"\nMerge Iteration {i+1}/{num_merges}")

    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge.")
        break

    sorted_pairs = sorted(pair_stats.items(), key=lambda item: item[1], reverse=True)
    print(f"Top 5 Pair Frequencies: {sorted_pairs[:5]}")

    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Found Best Pair: {best_pair} with Frequency: {best_freq}")

    current_splits = merge_pair(best_pair, current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merging {best_pair} into {new_token}")
    print(f"Splits after merge: {current_splits}")
    
    vocab.append(new_token)
    print(f"Updated Vocabular: {vocab}")

    merges[best_pair] = new_token
    print(f"Updated Merges: {merges}")

    print("-" * 30)

    print("\n--- BPE Merges Complete ---")
    print(f"Final Vocabulary Size: {len(vocab)}")
    print("\nLearned Merges (Pair -> New Token):")

    for pair, token in merges.items():
        print(f"{pair} -> {token}")

    print("\nFinal Word Splits: after all merges: ")
    print(current_splits)

    print("\nFinal Word Splits after all merges: ")
    print(current_splits)

    print("\nFinal Vocabulary (sorted):")
    final_vocab_sorted = sorted(list(set(vocab)))
    print(final_vocab_sorted)
