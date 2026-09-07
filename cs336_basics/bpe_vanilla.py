from collections import defaultdict
import regex as r
import os
from typing import BinaryIO
from multiprocessing import Pool

INPUT_FILE_PATH = "/Users/yuroc/workspace/learn/assignment1-basics/cs336_basics/inputs/TinyStoriesV2-GPT4-valid.txt"


class BPEVanilla:
    # simple implementation of BPE
    def __init__(self, vocab_size, input_path=None, specials: list[str] = [], end_of_doc: str = None):
        self.PAT: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab: dict[int, bytes] = {}  # token_id: bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
        if specials:
            for i, s in enumerate(specials):
                self.vocab[256 + i] = s.encode("utf-8")

        self.merge_rules: list[tuple[int]] = []  # use token_id [(1, 2, 3)]
        self.training_file_path: str | None = None
        self.pretoken_counts: defaultdict[tuple[int], int] = defaultdict(int)  # id
        self.new_vocab_idx = 256 + len(specials if specials else [])
        self.vocab_size = vocab_size
        self.input_path = input_path
        self.special_tokens = specials
        if end_of_doc:
            self.end_of_doc = end_of_doc
        elif specials:
            if "<|endoftext|>" in specials:
                self.end_of_doc = "<|endoftext|>"
            else:
                self.end_of_doc = specials[0]
        else:
            self.end_of_doc = "<|endoftext|>"

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

    def pretoken_count_one_chunk(self, input_path, s, e) -> defaultdict[tuple[int], int]:
        local_pretoken_counts: defaultdict[tuple[int], int] = defaultdict(int)  # id
        with open(input_path, "rb") as f:
            f.seek(s)
            text = f.read(e - s).decode("utf-8", errors="ignore")
            if self.special_tokens:
                patterns = "|".join(r.escape(sp) for sp in self.special_tokens)
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

    def get_initial_pretoken_counts_from_file(self, input_path: str) -> defaultdict[tuple[int], int]:
        if not input_path:
            raise Exception("input file/path is not provided")

        boundaries = []
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = self.find_chunk_boundaries(f, num_processes, self.end_of_doc.encode("utf-8"))

        bs = zip(boundaries[:-1], boundaries[1:])

        inputs = [(input_path, b[0], b[1]) for b in bs]
        with Pool(4) as p:
            results = p.starmap(self.pretoken_count_one_chunk, inputs)
        for res in results:
            for key, c in res.items():
                self.pretoken_counts[key] += c

    def get_pair_counts_from_pretoken(self):
        pair_counts: defaultdict[tuple[int], int] = defaultdict(int)
        max_count = 0
        pair_bytes_max_count = None
        pair_token_ids_max_count = None
        for word, c in self.pretoken_counts.items():
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += c

        max_count = max(pair_counts.values())
        for pair, c in pair_counts.items():
            if c == max_count:
                pair_bytes = (self.vocab[pair[0]], self.vocab[pair[1]])
                if pair_bytes_max_count is None or pair_bytes > pair_bytes_max_count:
                    pair_bytes_max_count: tuple[int, int] = pair_bytes
                    pair_token_ids_max_count: tuple[int, int] = pair

        return max_count, pair_token_ids_max_count, pair_counts

    def merge(self, token_id1, token_id2):
        # update self.vocab
        new_token_id = self.new_vocab_idx
        self.vocab[new_token_id] = self.vocab[token_id1] + self.vocab[token_id2]
        # update self.merge_rules
        self.merge_rules.append((token_id1, token_id2, new_token_id))
        # update self.pretoken_counts
        new_pretoken_counts = defaultdict(int)
        for pretoken, c in self.pretoken_counts.items():
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if pretoken[i] == token_id1 and i + 1 < len(pretoken) and pretoken[i + 1] == token_id2:
                    new_pretoken.append(new_token_id)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            new_pretoken_counts[tuple(new_pretoken)] += c
        self.pretoken_counts = new_pretoken_counts
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
            max_count, pair_max_count, pair_counts = self.get_pair_counts_from_pretoken()
            self.merge(pair_max_count[0], pair_max_count[1])

    def encode(self, text):
        token_id_list = text.encode("utf-8")
        for rule in self.merge_rules:
            a, b, c = rule[0], rule[1], rule[2]
            token_id_list = self.encode_merge_pair(token_id_list, a, b, c)
        return token_id_list

    def decode(self, byte_list):
        ans = []
        for t_id in byte_list:
            ans.append(self.vocab[t_id])
        s = ""
        for b in ans:
            s += b.decode("utf-8")
        return s


def main():
    bpe = BPEVanilla(1000, input_path=INPUT_FILE_PATH)
    bpe.train()
    print(bpe.decode(bpe.encode("i love you")))


if __name__ == "__main__":
    main()
