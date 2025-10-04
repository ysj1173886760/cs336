import regex
from collections import defaultdict
from copy import deepcopy
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from heapdict import heapdict
from functools import wraps
from typing import Iterable, Iterator
from tqdm import tqdm
import cProfile
import random
import heapq
import numpy as np


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} executed in {end - start:.6f} seconds")
        return result

    return wrapper


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

        self.split_by_special_re = regex.compile(
            "|".join(map(regex.escape, special_tokens))
        )

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
                    if (index < len(text_token) - 1) and (
                        text_token[index],
                        text_token[index + 1],
                    ) == candidate_pair:
                        new_sequence.append(text_token[index] + text_token[index + 1])
                        index += 2
                    else:
                        new_sequence.append(text_token[index])
                        index += 1
                new_occur_dict[tuple(new_sequence)] = freq

            self.time_statictics["calc_occur_dict"] += time.time() - last_time

            occur_dict = new_occur_dict

        print(f"merges with {merge_count} times, {time.time() - start_time}")

    def calc_candidiate_pair_by_occur_dict(self, occur_pair_dict):
        candidate_pair = ()
        candidate_freq = 0
        for pair, freq in occur_pair_dict.items():
            if freq > candidate_freq:
                candidate_freq = freq
                candidate_pair = pair
            elif freq == candidate_freq and pair > candidate_pair:
                candidate_pair = pair
        return candidate_pair, candidate_freq

    def get_sort_key(self, freq: int, pair: tuple):
        # 每个字节做 255 - x 反向，最后追加一个“超范围哨兵” 256
        # 这样就能把“前缀时，长的更大”的规则反向为“长的更小”（最小堆里更优先）
        return (
            -freq,
            tuple(255 - x for x in pair[0]) + (256,),
            tuple(255 - x for x in pair[1]) + (256,),
        )

    @timing
    def train(self, occur_dict: dict[tuple[bytes], int]):
        occur_pair_dict: dict[tuple[bytes, bytes], int] = defaultdict(int)
        pair_to_word_dict: dict[tuple[bytes, bytes], set] = defaultdict(
            lambda: defaultdict(int)
        )

        # remap occur dict
        id_to_sequence_dict: dict[int, tuple[bytes]] = {
            idx: sequence for idx, sequence in enumerate(occur_dict.keys())
        }
        occur_dict_new: dict[int, int] = {
            id: occur_dict[sequence] for id, sequence in id_to_sequence_dict.items()
        }
        occur_dict: dict[int, int] = occur_dict_new

        last_time = time.time()
        for seq_id, freq in occur_dict.items():
            byte_sequence = id_to_sequence_dict[seq_id]
            for i in range(len(byte_sequence) - 1):
                occur_pair_dict[(byte_sequence[i], byte_sequence[i + 1])] += freq
                pair_to_word_dict[(byte_sequence[i], byte_sequence[i + 1])][seq_id] += 1

        freq_heap = heapdict()
        for pair, freq in occur_pair_dict.items():
            freq_heap[pair] = self.get_sort_key(freq, pair)

        print(f"initial calc {time.time() - last_time}")

        assert len(self.history_merges) == 0

        merge_count = self.vocab_size - len(self.vocab)
        # merge_count = min(merge_count, 100)

        for epoch in tqdm(range(merge_count)):
            last_time = time.time()
            candidate_pair, candidate_freq = freq_heap.peekitem()
            self.time_statictics["calc_candidate"] += time.time() - last_time

            # check correctness
            # result_from_brute_force = self.calc_candidiate_pair_by_occur_dict(occur_pair_dict)
            # if result_from_brute_force[0] != candidate_pair or result_from_brute_force[1] != (-candidate_freq[0]):
            #   print(f"{result_from_brute_force}, {candidate_pair}, {(-candidate_freq[0])}")
            #   raise

            self.vocab[len(self.vocab)] = candidate_pair[0] + candidate_pair[1]
            self.history_merges.append((candidate_pair[0], candidate_pair[1]))

            # print(f"occur dict {occur_dict}")
            # print(f"merge {candidate_pair}")

            last_time = time.time()
            # update occur dict
            word_id_list = list(pair_to_word_dict[candidate_pair].keys())
            for word_id in word_id_list:
                freq = occur_dict[word_id]
                byte_sequence = id_to_sequence_dict[word_id]
                # generate new sequence
                index = 0
                old_change_position = []
                new_change_position = []
                new_sequence = []
                while index < len(byte_sequence):
                    if (index < len(byte_sequence) - 1) and (
                        byte_sequence[index],
                        byte_sequence[index + 1],
                    ) == candidate_pair:
                        old_change_position.append(index)
                        new_change_position.append(len(new_sequence))
                        new_sequence.append(
                            byte_sequence[index] + byte_sequence[index + 1]
                        )
                        index += 2
                    else:
                        new_sequence.append(byte_sequence[index])
                        index += 1
                new_sequence = tuple(new_sequence)
                id_to_sequence_dict[word_id] = new_sequence

                # calc changed pair
                # for every pair, change index - 1, index, index + 1
                pairs_to_remove = []
                pairs_to_add = []
                for index in old_change_position:
                    pairs_to_remove.append(
                        (byte_sequence[index], byte_sequence[index + 1])
                    )
                    if index > 0:
                        pairs_to_remove.append(
                            (byte_sequence[index - 1], byte_sequence[index])
                        )
                    if index + 2 < len(byte_sequence):
                        pairs_to_remove.append(
                            (byte_sequence[index + 1], byte_sequence[index + 2])
                        )
                for index in new_change_position:
                    if index > 0:
                        pairs_to_add.append(
                            (new_sequence[index - 1], new_sequence[index])
                        )
                    if index + 1 < len(new_sequence) and (
                        new_sequence[index] != new_sequence[index + 1]
                    ):
                        pairs_to_add.append(
                            (new_sequence[index], new_sequence[index + 1])
                        )

                for pair in pairs_to_remove:
                    pair_to_word_dict[pair][word_id] -= 1
                    if pair_to_word_dict[pair][word_id] == 0:
                        del pair_to_word_dict[pair][word_id]
                    occur_pair_dict[pair] -= freq
                    (origin_freq, bytes0, bytes1) = freq_heap[pair]
                    freq_heap[pair] = (-occur_pair_dict[pair], bytes0, bytes1)

                for pair in pairs_to_add:
                    pair_to_word_dict[pair][word_id] += 1
                    occur_pair_dict[pair] += freq
                    if pair in freq_heap:
                        (origin_freq, bytes0, bytes1) = freq_heap[pair]
                        freq_heap[pair] = (
                            -occur_pair_dict[pair],
                            bytes0,
                            bytes1,
                        )
                    else:
                        # calc new pair
                        freq_heap[pair] = self.get_sort_key(occur_pair_dict[pair], pair)

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

    @timing
    def train_from_scratch(
        self,
        path: str,
        process_num: int,
        chunk_num: int = 16,
        streaming_mode: bool = False,
    ):
        occur_dict: dict[tuple[bytes], int] = defaultdict(int)

        last_time = time.time()
        chunk_iter = BPETokenizerTrainer.read_from_path(path, chunk_num=chunk_num)
        chunks = []
        for chunk in chunk_iter:
            chunks.append(chunk)
        print(f"io consumption {time.time() - last_time}")

        last_time = time.time()
        # pre tokenization and init initial occurency dict

        def pretokenize_batch(chunks: list[str]):
            with ProcessPoolExecutor(max_workers=process_num) as pool:
                futures = [
                    pool.submit(
                        BPETokenizerTrainer.single_process_pretokenization,
                        chunk,
                        self.tok_re,
                        self.split_by_special_re,
                    )
                    for chunk in chunks
                ]
                for f in as_completed(futures):
                    for sequence, freq in f.result().items():
                        occur_dict[sequence] += freq

        if streaming_mode:
            # batch submit, may not much efficient
            for i in tqdm(range(0, len(chunks), process_num)):
                pretokenize_batch(chunks[i : i + process_num])
        else:
            pretokenize_batch(chunks)

        self.time_statictics["pre_tokenize"] += time.time() - last_time
        print(f"pre tokenize {time.time() - last_time}")

        # profiler = cProfile.Profile()
        # profiler.enable()

        self.train(occur_dict)

        # profiler.disable()
        # profiler.dump_stats("profile.out")

    def print_time_statictics(self):
        print(f"print time statistics")
        for name, time_consumption in self.time_statictics.items():
            print(f"{name} {time_consumption}")

    def save_to(self, path: str):
        print(f"save tokenizer to path: {path}")
        result_dict = {"vocab": self.vocab, "merges": self.history_merges}
        with open(path, "wb") as f:
            pickle.dump(result_dict, f)

    @staticmethod
    def read_from_path(file_path: str, chunk_num: int = 1) -> Iterator[str]:
        with open(file_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                yield chunk


class BPETokenizer:
    merges: list[tuple[bytes, bytes]]
    vocab: dict[int, bytes]
    special_token: list[str] | None

    vocab_reverse: dict[bytes, int]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab, merges, special_token=None):
        self.vocab = vocab
        self.merges = merges
        self.special_token = special_token

        if self.special_token is not None:
            self.special_token = sorted(self.special_token, key=len, reverse=True)
            pattern = "(" + "|".join(map(regex.escape, self.special_token)) + ")"
            self.split_by_special_re = regex.compile(pattern)
            self.special_token_set = set(self.special_token)
        else:
            self.split_by_special_re = None

        self.vocab_reverse = {}
        max_token_id = 0
        for token_id, bytes in self.vocab.items():
            self.vocab_reverse[bytes] = token_id
            max_token_id = max(token_id, max_token_id)

        # add special token
        if self.special_token is not None:
            for token in self.special_token:
                encoded_token = token.encode("utf-8")
                if encoded_token not in self.vocab_reverse:
                    max_token_id += 1
                    self.vocab[max_token_id] = encoded_token
                    self.vocab_reverse[encoded_token] = max_token_id

        self.pre_tokenize_re = regex.compile(self.PAT)
        self.time_statistics = defaultdict(int)

        self.merge_dict = {merge: i for i, merge in enumerate(self.merges)}

    @staticmethod
    def from_files(file_path, special_tokens=None) -> "BPETokenizer":
        with open(file_path, "rb") as f:
            result_dict = pickle.load(f)

        vocab = result_dict["vocab"]
        merges = result_dict["merges"]
        return BPETokenizer(vocab, merges, special_tokens)

    def pretokenize(self, text: str) -> list[tuple[bytes] | int]:
        result_seq = []
        chunks = []
        if self.split_by_special_re is not None:
            chunks = self.split_by_special_re.split(text)
            # encode special token directly
            for i in range(len(chunks)):
                if chunks[i] in self.special_token_set:
                    chunks[i] = self.vocab_reverse[chunks[i].encode("utf-8")]
        else:
            chunks = [text]

        for chunk in chunks:
            if isinstance(chunk, int):
                result_seq.append(chunk)
            else:
                for m in self.pre_tokenize_re.finditer(chunk):
                    token_bytes = m.group(0).encode("utf-8")
                    byte_sequence = tuple([token.to_bytes() for token in token_bytes])
                    result_seq.append(byte_sequence)

        return result_seq

    def tokenize_v3(self, byte_sequence: tuple[bytes]) -> list[int]:
        pair_set = defaultdict(int)
        for i in range(len(byte_sequence) - 1):
            pair_set[(byte_sequence[i], byte_sequence[i + 1])] += 1

        def remove_pair(pair):
            cnt = pair_set.get(pair)
            if cnt is None:
                return
            elif cnt == 1:
                del pair_set[pair]
            else:
                pair_set[pair] -= 1

        heap = []

        def add_pair_to_heap(pair):
            # only insert valid pair
            rank = self.merge_dict.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, pair))

        for pair in pair_set.keys():
            add_pair_to_heap(pair)

        while heap:
            rank, merge = heapq.heappop(heap)
            if merge not in pair_set:
                continue

            index = 0
            old_change_position = []
            new_change_position = []
            new_sequence = []
            while index < len(byte_sequence):
                if (index < len(byte_sequence) - 1) and (
                    byte_sequence[index],
                    byte_sequence[index + 1],
                ) == merge:
                    old_change_position.append(index)
                    new_change_position.append(len(new_sequence))
                    new_sequence.append(byte_sequence[index] + byte_sequence[index + 1])
                    index += 2
                else:
                    new_sequence.append(byte_sequence[index])
                    index += 1
            new_sequence = tuple(new_sequence)

            # calc changed pair
            # for every pair, change index - 1, index, index + 1
            for index in old_change_position:
                remove_pair((byte_sequence[index], byte_sequence[index + 1]))
                if index > 0:
                    remove_pair((byte_sequence[index - 1], byte_sequence[index]))
                if index + 2 < len(byte_sequence):
                    remove_pair((byte_sequence[index + 1], byte_sequence[index + 2]))

            for index in new_change_position:
                if index > 0:
                    pair_set[(new_sequence[index - 1], new_sequence[index])] += 1
                    add_pair_to_heap((new_sequence[index - 1], new_sequence[index]))
                if index + 1 < len(new_sequence) and (
                    new_sequence[index] != new_sequence[index + 1]
                ):
                    pair_set[(new_sequence[index], new_sequence[index + 1])] += 1
                    add_pair_to_heap((new_sequence[index], new_sequence[index + 1]))

            byte_sequence = new_sequence

        return [self.vocab_reverse[bytes] for bytes in byte_sequence]

    def tokenize_v2(self, byte_sequence: tuple[bytes]) -> list[int]:
        pair_set = defaultdict(int)
        for i in range(len(byte_sequence) - 1):
            pair_set[(byte_sequence[i], byte_sequence[i + 1])] += 1

        def remove_pair(pair):
            pair_set[pair] -= 1
            if pair_set[pair] == 0:
                del pair_set[pair]

        for merge in self.merges:
            if merge not in pair_set:
                continue

            print(f"merge: {merge}")
            print(pair_set)

            index = 0
            old_change_position = []
            new_change_position = []
            new_sequence = []
            while index < len(byte_sequence):
                if (index < len(byte_sequence) - 1) and (
                    byte_sequence[index],
                    byte_sequence[index + 1],
                ) == merge:
                    old_change_position.append(index)
                    new_change_position.append(len(new_sequence))
                    new_sequence.append(byte_sequence[index] + byte_sequence[index + 1])
                    index += 2
                else:
                    new_sequence.append(byte_sequence[index])
                    index += 1
            new_sequence = tuple(new_sequence)

            # calc changed pair
            # for every pair, change index - 1, index, index + 1
            for index in old_change_position:
                remove_pair((byte_sequence[index], byte_sequence[index + 1]))
                if index > 0:
                    remove_pair((byte_sequence[index - 1], byte_sequence[index]))
                if index + 2 < len(byte_sequence):
                    remove_pair((byte_sequence[index + 1], byte_sequence[index + 2]))
            for index in new_change_position:
                if index > 0:
                    pair_set[(new_sequence[index - 1], new_sequence[index])] += 1
                if index + 1 < len(new_sequence) and (
                    new_sequence[index] != new_sequence[index + 1]
                ):
                    pair_set[(new_sequence[index], new_sequence[index + 1])] += 1

            byte_sequence = new_sequence
        return [self.vocab_reverse[bytes] for bytes in byte_sequence]

    def tokenize(self, byte_sequence: tuple[bytes]) -> list[int]:
        bytes_set = set(byte_sequence)
        for merge in self.merges:
            if (merge[0] not in bytes_set) or (merge[1] not in bytes_set):
                continue

            new_sequence = []

            index = 0
            found_match: bool = False

            while index < len(byte_sequence):
                if (index < len(byte_sequence) - 1) and (
                    byte_sequence[index],
                    byte_sequence[index + 1],
                ) == merge:
                    if found_match == False:
                        found_match = True
                        for i in range(index):
                            new_sequence.append(byte_sequence[i])

                    new_sequence.append(byte_sequence[index] + byte_sequence[index + 1])
                    index += 2
                else:
                    if found_match:
                        new_sequence.append(byte_sequence[index])
                    index += 1

            if found_match:
                byte_sequence = new_sequence
                bytes_set = set(byte_sequence)

        # encode id
        result = []
        for bytes in byte_sequence:
            result.append(self.vocab_reverse[bytes])

        return result

    def encode(self, text: str) -> list[int]:
        chunks = self.pretokenize(text)

        # for every single chunk, perform merge
        result = []
        for chunk in chunks:
            if isinstance(chunk, int):
                result.append(chunk)
            else:
                # result.extend(self.tokenize(chunk))
                # result.extend(self.tokenize_v2(chunk))
                result.extend(self.tokenize_v3(chunk))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token

    def decode(self, ids: list[int]) -> str:
        result_byte = bytes()
        for id in ids:
            result_byte += self.vocab[id]

        return result_byte.decode("utf-8", errors="replace")


def naive_test():
    raw_str = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    trainer = BPETokenizerTrainer(270, ["<|endoftext|>"])
    result = trainer.train(raw_str)


def train_tiny_stories(process_num: int = 16):
    valid_path = (
        "/Users/bytedance/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    )
    train_path = (
        "/Users/bytedance/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    )
    # path = valid_path
    path = train_path
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    trainer = BPETokenizerTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    trainer.train_from_scratch(path, process_num)

    trainer.print_time_statictics()

    trainer.save_to("tiny_stories_result.pkl")


def train_open_web_text(process_num: int = 16, chunk_num: int = 128):
    train_path = "/Users/bytedance/cs336/assignment1-basics/data/owt_train.txt"
    valid_path = "/Users/bytedance/cs336/assignment1-basics/data/owt_valid.txt"
    # path = valid_path
    path = train_path
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    trainer = BPETokenizerTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    trainer.train_from_scratch(
        path, process_num, chunk_num=chunk_num, streaming_mode=True
    )

    trainer.print_time_statictics()

    trainer.save_to("open_web_text_result.pkl")


def split_and_sample_data(text: str, sample_num: int = 10) -> list[str]:
    special_tokens: list[str] = ["<|endoftext|>"]
    split_by_special_re = regex.compile("|".join(map(regex.escape, special_tokens)))
    chunks = split_by_special_re.split(text)
    return random.sample(chunks, sample_num)


def calc_compression_ratio(dataset_path, tokenizer_path, sample_count=10):
    tokenizer = BPETokenizer.from_files(tokenizer_path)

    chunks = [
        chunk for chunk in BPETokenizerTrainer.read_from_path(dataset_path, chunk_num=1)
    ]
    sample_docs = split_and_sample_data(chunks[0], sample_count)

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    token_length = sum(len(tokenizer.encode(doc)) for doc in sample_docs)
    end_time = time.time()

    profiler.disable()
    profiler.dump_stats("profile.out")

    bytes_length = sum(len(doc.encode("utf-8")) for doc in sample_docs)

    print(
        f"token_length: {token_length}. bytes_length: {bytes_length}. compression_ratio: {token_length / bytes_length}."
    )
    print(
        f"tokenize time {end_time - start_time}. estimate throughput: {bytes_length / (end_time - start_time)} bytes/second"
    )
    print(tokenizer.time_statistics)


def tokenize_data(data_path, tokenizer_path, output_path):
    tokenizer = BPETokenizer.from_files(tokenizer_path)
    chunk_num = 1024
    batch_size = 16
    process_num = 9

    def encode_parallel(chunks: list[str]):
        with ProcessPoolExecutor(max_workers=process_num) as pool:
            futures = [pool.submit(tokenizer.encode, chunk) for chunk in chunks]
            for f in as_completed(futures):
                tokens = f.result()
                array = np.array(tokens, dtype=np.uint16)
                final_result.append(array)

    final_result = []
    chunks = [
        chunk
        for chunk in BPETokenizerTrainer.read_from_path(data_path, chunk_num=chunk_num)
    ]

    for i in tqdm(range(0, len(chunks), batch_size)):
        encode_parallel(chunks[i : i + batch_size])

    final_result = np.concatenate(final_result)
    np.save(output_path, final_result)


if __name__ == "__main__":
    # train_tiny_stories()
    # train_open_web_text(process_num=8, chunk_num=128)

    # calc_compression_ratio(
    #     "/Users/bytedance/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
    #     "tiny_stories_result.pkl",
    #     sample_count=1000,
    # )
    # calc_compression_ratio("/Users/bytedance/cs336/assignment1-basics/data/owt_valid.txt", "open_web_text_result.pkl")
    # calc_compression_ratio("/Users/bytedance/cs336/assignment1-basics/data/owt_valid.txt", "tiny_stories_result.pkl")
    tokenize_data(
        "/Users/bytedance/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        "tiny_stories_result.pkl",
        "tiny_stories_train.npy",
    )
    tokenize_data(
        "/Users/bytedance/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
        "tiny_stories_result.pkl",
        "tiny_stories_valid.npy",
    )
