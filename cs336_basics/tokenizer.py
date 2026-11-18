import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_word(word, pair):
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(input_path, start, end, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    chunks = re.split("|".join(re.escape(t) for t in special_tokens), text) if special_tokens else [text]
    word_counts = Counter()
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            word_counts[tuple(bytes([b]) for b in m.group().encode("utf-8"))] += 1
    return word_counts


def train_bpe(input_path, vocab_size, special_tokens, num_processes=8):
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    with Pool(num_processes) as pool:
        results = pool.starmap(
            pretokenize,
            [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])],
        )
    word_counts = Counter()
    for counts in results:
        word_counts.update(counts)

    pair_counts = defaultdict(int)
    pair_words = defaultdict(set)
    for word, count in word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += count
            pair_words[pair].add(word)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        for word in list(pair_words[best]):
            count = word_counts.pop(word)
            for pair in zip(word, word[1:]):
                pair_counts[pair] -= count
                pair_words[pair].discard(word)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
            new_word = merge_word(word, best)
            word_counts[new_word] += count
            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += count
                pair_words[pair].add(new_word)

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath) as f:
            vocab = {int(k): bytes.fromhex(v) for k, v in json.load(f).items()}
        with open(merges_filepath) as f:
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in json.load(f)]
        return cls(vocab, merges, special_tokens)

    def encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        tokens = [bytes([b]) for b in word]
        while len(tokens) > 1:
            rank, i = min(
                (self.merge_ranks.get(pair, float("inf")), i) for i, pair in enumerate(zip(tokens, tokens[1:]))
            )
            if rank == float("inf"):
                break
            tokens[i : i + 2] = [tokens[i] + tokens[i + 1]]
        self.cache[word] = [self.token_to_id[t] for t in tokens]
        return self.cache[word]

    def encode(self, text):
        if self.special_tokens:
            parts = re.split("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")", text)
        else:
            parts = [text]
        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                for m in re.finditer(PAT, part):
                    ids.extend(self.encode_word(m.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
