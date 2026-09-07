import os
from collections import Counter, defaultdict
from itertools import pairwise
from multiprocessing import Pool
from typing import BinaryIO

import regex as r

INPUT_FILE_PATH = "/Users/yuroc/workspace/learn/assignment1-basics/cs336_basics/inputs/TinyStoriesV2-GPT4-valid.txt"


class BPE:
    def __init__(
        self,
        vocab_size: int,
        input_path: str | None = None,
        specials: list[str] | None = None,
        end_of_doc: str | None = None,
    ):
        specials = list(specials or [])
        if len(set(specials)) != len(specials):
            raise ValueError("special tokens must be unique")
        if any(not special for special in specials):
            raise ValueError("special tokens must not be empty")
        if vocab_size < 256 + len(specials):
            raise ValueError("vocab_size must include all 256 byte tokens and special tokens")

        self.PAT: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab: dict[int, bytes] = {}  # token_id: bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
        for i, special in enumerate(specials):
            self.vocab[256 + i] = special.encode("utf-8")

        self.merge_rules: list[tuple[int, int, int]] = []
        self.training_file_path: str | None = None
        self.pretoken_counts: defaultdict[tuple[int, ...], int] = defaultdict(
            int
        )  # {(token_idx1, token_idx2, token_idx3, ...): count)}

        self.pretoken_counts_with_index: defaultdict[int, tuple[int, ...]] = defaultdict(
            tuple
        )  # {pretoken_idx:(token_idx1, token_idx2, token_idx3, ..., count)}
        self.initial_vocab_size = 256 + len(specials)
        self.new_vocab_idx = self.initial_vocab_size
        self.vocab_size = vocab_size
        self.input_path = input_path
        self.special_tokens = specials
        self.special_token_to_id = {special: 256 + i for i, special in enumerate(specials)}
        if end_of_doc:
            self.end_of_doc = end_of_doc
        elif specials:
            if "<|endoftext|>" in specials:
                self.end_of_doc = "<|endoftext|>"
            else:
                self.end_of_doc = specials[0]
        else:
            self.end_of_doc = "<|endoftext|>"
        self.pair_buckets = {}  # {count: set[(token_id_a, token_id_b, byte_a, byte_b)]}
        self.pair_count = {}  # {(p1, p2): [c, set(pretoken_idx)]}
        self.max_pair_count = 0

    def _reset_training_state(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        for special, token_id in self.special_token_to_id.items():
            self.vocab[token_id] = special.encode("utf-8")
        self.merge_rules.clear()
        self.new_vocab_idx = self.initial_vocab_size
        self.pretoken_counts.clear()
        self.pretoken_counts_with_index.clear()
        self.pair_count.clear()
        self.pair_buckets.clear()
        self.max_pair_count = 0

    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
        if not split_special_token:
            raise ValueError("split_special_token must not be empty")

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
            overlap = b""
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                search_chunk = overlap + mini_chunk
                found_at = search_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = file.tell() - len(search_chunk) + found_at
                    break
                overlap_size = len(split_special_token) - 1
                overlap = search_chunk[-overlap_size:] if overlap_size else b""

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def pretoken_count_one_chunk(self, input_path, s, e) -> defaultdict[tuple[int, ...], int]:
        local_pretoken_counts: defaultdict[tuple[int, ...], int] = defaultdict(int)  # id
        with open(input_path, "rb") as f:
            f.seek(s)
            text = f.read(e - s).decode("utf-8", errors="ignore")
            if self.special_tokens:
                patterns = "|".join(r.escape(sp) for sp in sorted(self.special_tokens, key=len, reverse=True))
                mini_chunks = r.split(patterns, text)
                for chk in mini_chunks:
                    mini_chunk_counts = defaultdict(int)
                    matches = r.finditer(self.PAT, chk)
                    for match in matches:
                        pretoken = tuple(match.group().encode("utf-8"))
                        mini_chunk_counts[pretoken] += 1
                    for k, v in mini_chunk_counts.items():
                        local_pretoken_counts[k] += v
            else:
                matches = r.finditer(self.PAT, text)
                for match in matches:
                    pretoken = tuple(match.group().encode("utf-8"))
                    local_pretoken_counts[pretoken] += 1

        return local_pretoken_counts

    def get_initial_pretoken_counts_from_file(self, input_path: str) -> defaultdict[tuple[int, ...], int]:
        if not input_path:
            raise ValueError("input file/path is not provided")

        self._reset_training_state()

        boundaries = []
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, self.end_of_doc.encode("utf-8"))

        bs = pairwise(boundaries)

        inputs = [(input_path, b[0], b[1]) for b in bs]
        with Pool(4) as p:
            results = p.starmap(self.pretoken_count_one_chunk, inputs)
        for res in results:
            for key, c in res.items():
                self.pretoken_counts[key] += c
        # convert to pretoken_counts_with_index for merging optimization
        for pretoken_idx, (t, c) in enumerate(self.pretoken_counts.items()):
            # initial counting and finding max
            self.pretoken_counts_with_index[pretoken_idx] = t + (c,)
        self.initial_pair_counting()
        return self.pretoken_counts

    def initial_pair_counting(self):
        # initial pair counting and construct the pair heap
        for pretoken_idx, t in self.pretoken_counts_with_index.items():
            pretokens, c = t[:-1], t[-1]
            for a, b in pairwise(pretokens):
                if (a, b) in self.pair_count:
                    self.pair_count[(a, b)][0] += c
                    self.pair_count[(a, b)][1].add(pretoken_idx)
                else:
                    self.pair_count[(a, b)] = [c, {pretoken_idx}]
        # self.pair_count is ready, now construct self.pair_bucket
        for k, v in self.pair_count.items():
            a, b = k
            c = v[0]
            self.max_pair_count = max(self.max_pair_count, c)
            if c not in self.pair_buckets:
                self.pair_buckets[c] = {(self.vocab[a], self.vocab[b], a, b)}
            else:
                self.pair_buckets[c].add((self.vocab[a], self.vocab[b], a, b))

    def get_max_pair_from_bucket(self):
        while self.max_pair_count > 0 and (
            (self.max_pair_count not in self.pair_buckets) or not self.pair_buckets[self.max_pair_count]
        ):
            self.max_pair_count -= 1
        if self.max_pair_count == 0:
            return None
        return max(self.pair_buckets[self.max_pair_count])

    def _bucket_entry(self, pair: tuple[int, int]) -> tuple[bytes, bytes, int, int]:
        a, b = pair
        return self.vocab[a], self.vocab[b], a, b

    def _remove_from_bucket(self, pair: tuple[int, int], count: int) -> None:
        bucket = self.pair_buckets.get(count)
        if bucket is None:
            return
        bucket.discard(self._bucket_entry(pair))
        if not bucket:
            del self.pair_buckets[count]

    def _update_pair_count(
        self,
        pair: tuple[int, int],
        count_delta: int,
        pretoken_idx: int,
        remains_in_pretoken: bool,
    ) -> None:
        pair_info = self.pair_count.get(pair)
        if pair_info is None:
            old_count = 0
            indices = set()
        else:
            old_count = pair_info[0]
            indices = pair_info[1]

        if old_count:
            self._remove_from_bucket(pair, old_count)

        new_count = old_count + count_delta
        if new_count < 0:
            raise RuntimeError(f"negative count produced for pair {pair}")

        if remains_in_pretoken:
            indices.add(pretoken_idx)
        else:
            indices.discard(pretoken_idx)

        if new_count == 0:
            self.pair_count.pop(pair, None)
            return

        if pair_info is None:
            self.pair_count[pair] = [new_count, indices]
        else:
            pair_info[0] = new_count
        self.pair_buckets.setdefault(new_count, set()).add(self._bucket_entry(pair))
        self.max_pair_count = max(self.max_pair_count, new_count)

    def get_max_pair_and_merge(self):
        # self.pair_buckets[self.max_pair_count] should have the max count already
        max_pair = self.get_max_pair_from_bucket()
        if max_pair is None:
            raise ValueError("cannot grow vocabulary: the corpus has no mergeable token pairs")
        byte_a, byte_b, token_id_a, token_id_b = max_pair
        # so now we need to merge token_id_a, and token_id_b
        # first create new token id and add it the vocab
        new_token_id = self.new_vocab_idx
        self.vocab[new_token_id] = byte_a + byte_b
        # update self.merge_rules
        self.merge_rules.append((token_id_a, token_id_b, new_token_id))
        # update self.pretoken_counts, only update the pretokens that the pair (a, b) occurred
        # 1. remove occurrance of (a, b) from pretoken_counts, and update the bucket and heap
        # 2. if there is item before (a, b), say (x, a, b), then reduce the counts of (x, a), add the count of (x, ab)
        # 3. if there is item after, say (a, b, y), then reduce the count of (b, y), and increase the count of (ab, y)
        pretoken_ids_occurred = set(self.pair_count[(token_id_a, token_id_b)][1])
        for pretoken_idx in pretoken_ids_occurred:
            tokens = self.pretoken_counts_with_index[pretoken_idx][:-1]
            pretoken_count = self.pretoken_counts_with_index[pretoken_idx][-1]
            new_tokens = self.encode_merge_pair(tokens, token_id_a, token_id_b, new_token_id)
            old_pairs = Counter(pairwise(tokens))
            new_pairs = Counter(pairwise(new_tokens))

            for pair in old_pairs.keys() | new_pairs.keys():
                count_delta = (new_pairs[pair] - old_pairs[pair]) * pretoken_count
                if count_delta:
                    self._update_pair_count(
                        pair,
                        count_delta,
                        pretoken_idx,
                        new_pairs[pair] > 0,
                    )

            self.pretoken_counts_with_index[pretoken_idx] = tuple(new_tokens) + (pretoken_count,)

        self.new_vocab_idx += 1

    def encode_merge_pair(self, input_token_id_list, p_a, p_b, c):
        # merge all occurence of the pair (p_a, p_b)
        i = 0
        new_list = []
        while i < len(input_token_id_list):
            if i + 1 < len(input_token_id_list) and input_token_id_list[i] == p_a and input_token_id_list[i + 1] == p_b:
                new_list.append(c)
                i += 2
            else:
                new_list.append(input_token_id_list[i])
                i += 1
        return new_list

    def train(self):
        self.get_initial_pretoken_counts_from_file(self.input_path)

        while len(self.vocab) < self.vocab_size:
            self.get_max_pair_and_merge()

    def _encode_ordinary_text(self, text: str) -> list[int]:
        encoded: list[int] = []
        for match in r.finditer(self.PAT, text):
            token_id_list: list[int] = list(match.group().encode("utf-8"))
            for a, b, c in self.merge_rules:
                token_id_list = self.encode_merge_pair(token_id_list, a, b, c)
            encoded.extend(token_id_list)
        return encoded

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_ordinary_text(text)

        # Longest-first ordering preserves overlapping special tokens.
        special_pattern = (
            "(" + "|".join(r.escape(special) for special in sorted(self.special_tokens, key=len, reverse=True)) + ")"
        )
        encoded: list[int] = []
        for chunk in r.split(special_pattern, text):
            if not chunk:
                continue
            special_token_id = self.special_token_to_id.get(chunk)
            if special_token_id is not None:
                encoded.append(special_token_id)
            else:
                encoded.extend(self._encode_ordinary_text(chunk))
        return encoded

    def decode(self, token_ids) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        return token_bytes.decode("utf-8", errors="replace")


def main():
    bpe = BPE(1000, input_path=INPUT_FILE_PATH)
    bpe.train()
    print(bpe.decode(bpe.encode("i love you")))


if __name__ == "__main__":
    main()
