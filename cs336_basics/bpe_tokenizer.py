import regex
from collections import defaultdict
from copy import deepcopy
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


class BPETokenizerTrainer:
  special_tokens: list[str] = ["<|endoftext|>"]
  vocab_size: int

  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

  history_merges: list[tuple[bytes, bytes]]
  vocab: dict[int, bytes]

  time_statictics = defaultdict(int)

  def __init__(self, vocab_size, special_tokens):
    self.vocab_size = vocab_size
    self.special_tokens = special_tokens
    self.tok_re = regex.compile(self.PAT)

    self.split_by_special_re = regex.compile("|".join(map(regex.escape, special_tokens)))

    self.vocab = {i: bytes([i]) for i in range(256)}
    self.history_merges = []
    self.time_statictics = defaultdict(int)
    
    for idx, special_token in enumerate(self.special_tokens):
      self.vocab[256 + idx] = special_token.encode("utf-8")
  
  def naive_train(self, text: str):
    start_time = time.time()

    occur_dict: dict[tuple[bytes], int] = defaultdict(int)

    # pre tokenization and init initial occurency dict
    # first split by special token
    chunks = self.split_by_special_re.split(text)
    for chunk in chunks:
      for m in self.tok_re.finditer(chunk):
        token_bytes = m.group(0).encode("utf-8")
        byte_sequence = tuple([token.to_bytes() for token in token_bytes])
        occur_dict[byte_sequence] += 1
    
    print(f"pre tokenize {time.time() - start_time}")
    start_time = time.time()
    
    merge_count = self.vocab_size - len(self.vocab)

    for epoch in range(merge_count):
      occur_pair_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)

      last_time = time.time()
      for text_token, freq in occur_dict.items():
        for i in range(len(text_token) - 1):
          occur_pair_dict[(text_token[i], text_token[i + 1])] += freq
      self.time_statictics["calc_freq"] += time.time() - last_time

      last_time = time.time()
      candidate_pair = ()
      candidate_freq = 0
      for pair, freq in occur_pair_dict.items():
        if freq > candidate_freq:
          candidate_freq = freq
          candidate_pair = pair
        elif freq == candidate_freq and pair > candidate_pair:
          candidate_pair = pair
      self.time_statictics["calc_candidate"] += time.time() - last_time

      self.vocab[len(self.vocab)] = candidate_pair[0] + candidate_pair[1]
      self.history_merges.append((candidate_pair[0], candidate_pair[1]))
      
      # re-calc occur_dict
      last_time = time.time()
      new_occur_dict = {}
      for text_token, freq in occur_dict.items():
        index = 0
        new_sequence = []
        while index < len(text_token):
          if (index < len(text_token) - 1) and (text_token[index], text_token[index + 1]) == candidate_pair:
            new_sequence.append(text_token[index] + text_token[index + 1])
            index += 2
          else:
            new_sequence.append(text_token[index])
            index += 1
        new_occur_dict[tuple(new_sequence)] = freq

      self.time_statictics["calc_occur_dict"] += time.time() - last_time

      occur_dict = new_occur_dict

    print(f"merges with {merge_count} times, {time.time() - start_time}")
  
  def train(self, occur_dict: dict[tuple[bytes], int]):
    occur_pair_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_word_dict: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)

    last_time = time.time()
    for byte_sequence, freq in occur_dict.items():
      for i in range(len(byte_sequence) - 1):
        occur_pair_dict[(byte_sequence[i], byte_sequence[i + 1])] += freq
        pair_to_word_dict[(byte_sequence[i], byte_sequence[i + 1])].add(byte_sequence)
    print(f"initial calc {time.time() - last_time}")

    assert len(self.history_merges) == 0

    merge_count = self.vocab_size - len(self.vocab)

    for epoch in range(merge_count):
      candidate_pair = ()
      candidate_freq = 0
      last_time = time.time()
      for pair, freq in occur_pair_dict.items():
        if freq > candidate_freq:
          candidate_freq = freq
          candidate_pair = pair
        elif freq == candidate_freq and pair > candidate_pair:
          candidate_pair = pair

      self.time_statictics["calc_candidate"] += time.time() - last_time

      self.vocab[len(self.vocab)] = candidate_pair[0] + candidate_pair[1]
      self.history_merges.append((candidate_pair[0], candidate_pair[1]))

      # print(f"occur dict {occur_dict}")
      # print(f"merge {candidate_pair}")
      
      last_time = time.time()
      # update occur dict
      word_list = deepcopy(pair_to_word_dict[candidate_pair])
      for byte_sequence in word_list:
        freq = occur_dict.pop(byte_sequence)
        # remove origin freq.
        for i in range(len(byte_sequence) - 1):
            occur_pair_dict[(byte_sequence[i], byte_sequence[i + 1])] -= freq
            pair_to_word_dict[(byte_sequence[i], byte_sequence[i + 1])].discard(byte_sequence)
        # generate new sequence
        index = 0
        new_sequence = []
        while index < len(byte_sequence):
          if (index < len(byte_sequence) - 1) and (byte_sequence[index], byte_sequence[index + 1]) == candidate_pair:
            new_sequence.append(byte_sequence[index] + byte_sequence[index + 1])
            index += 2
          else:
            new_sequence.append(byte_sequence[index])
            index += 1
        new_sequence = tuple(new_sequence)
        occur_dict[new_sequence] += freq

        for i in range(len(new_sequence) - 1):
          occur_pair_dict[(new_sequence[i], new_sequence[i + 1])] += freq
          pair_to_word_dict[(new_sequence[i], new_sequence[i + 1])].add(new_sequence)
      
      self.time_statictics["calc_occur_dict"] += time.time() - last_time
  
  @staticmethod
  def single_process_pretokenization(text: str, tok_re, split_by_special_re):
      occur_dict: dict[tuple[bytes], int] = defaultdict(int)
      chunks = split_by_special_re.split(text)
      for chunk in chunks:
        for m in tok_re.finditer(chunk):
          token_bytes = m.group(0).encode("utf-8")
          byte_sequence = tuple([token.to_bytes() for token in token_bytes])
          occur_dict[byte_sequence] += 1
      return occur_dict
  
  def train_from_scratch(self, path: str, process_num: int):
    occur_dict: dict[tuple[bytes], int] = defaultdict(int)

    last_time = time.time()
    chunks = BPETokenizerTrainer.read_from_path(path, process_num=process_num)
    print(f"io consumption {time.time() - last_time}")

    last_time = time.time()
    # pre tokenization and init initial occurency dict
    with ProcessPoolExecutor(max_workers=process_num) as pool:
      futures = [pool.submit(BPETokenizerTrainer.single_process_pretokenization, chunk, self.tok_re, self.split_by_special_re) for chunk in chunks]
      for f in as_completed(futures):
        for sequence, freq in f.result().items():
          occur_dict[sequence] += freq

    self.time_statictics["pre_tokenize"] += time.time() - last_time
    print(f"pre tokenize {time.time() - last_time}")
    
    self.train(occur_dict)

  def print_time_statictics(self):
    for name, time_consumption in self.time_statictics.items():
      print(f"{name} {time_consumption}")
    
  @staticmethod
  def read_from_path(file_path: str, process_num: int = 1) -> list[str]:
    chunk_list = []
    with open(file_path, "rb") as f:
      boundaries = find_chunk_boundaries(f, process_num, b"<|endoftext|>")

      # The following is a serial implementation, but you can parallelize this
      # by sending each start/end pair to a set of processes.
      for start, end in zip(boundaries[:-1], boundaries[1:]):
          f.seek(start)
          chunk = f.read(end - start).decode("utf-8", errors="ignore")
          # Run pre-tokenization on your chunk and store the counts for each pre-token
          chunk_list.append(chunk)

    return chunk_list
      

def naive_test():
  raw_str = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
  trainer = BPETokenizerTrainer(270, ["<|endoftext|>"])
  result = trainer.train(raw_str)

def train_tiny_stories(process_num: int = 16):
  valid_path = "/Users/bytedance/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
  train_path = "/Users/bytedance/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
  vocab_size = 10000
  special_tokens = ["<|endoftext|>"]

  trainer = BPETokenizerTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
  trainer.train_from_scratch(valid_path, process_num)

  trainer.print_time_statictics()

  with open("result.pkl", "wb") as f:
      pickle.dump(trainer.vocab, f)      # 任意对象（含 bytes）一步到

if __name__ == '__main__':
  train_tiny_stories()
  