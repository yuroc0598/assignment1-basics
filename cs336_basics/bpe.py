from collections import defaultdict
import regex as r

INPUT_FILE_PATH = "/Users/yuroc/workspace/learn/assignment1-basics/cs336_basics/inputs/TinyStoriesV2-GPT4-valid.txt"


class BPEVanilla:
    # simple implementation of BPE, without several pretokenization, training or merge rules
    def __init__(self, vocab_size, specials=None):
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

    def get_initial_pretoken_counts(self, text) -> defaultdict[tuple[int], int]:
        # {(byte, byte, byte): count}
        matches = r.finditer(self.PAT, text)
        for match in matches:
            pretoken = tuple(match.group().encode("utf-8"))
            self.pretoken_counts[pretoken] += 1

    def get_pair_counts_from_pretoken(self):
        pair_counts: defaultdict[tuple[int], int] = defaultdict(int)
        max_count = 0
        pair_max_count = None
        for word, c in self.pretoken_counts.items():
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += c
                if pair_counts[(a, b)] > max_count:
                    max_count = pair_counts[(a, b)]
                    pair_max_count: tuple[int, int] = (a, b)
        return max_count, pair_max_count, pair_counts

    def merge(self, token_id1, token_id2):
        # update self.vocab
        new_token_id = self.new_vocab_idx
        self.vocab[new_token_id] = self.vocab[token_id1] + self.vocab[token_id2]
        # update self.merge_rules
        self.merge_rules.append((token_id1, token_id2, new_token_id))
        # update self.pretoken_counts
        new_pretoken_counts = defaultdict(int)
        for pretoken, c in self.pretoken_counts.items():
            new_word = []
            i = 0
            while i < len(pretoken):
                if pretoken[i] == token_id1 and i + 1 < len(pretoken) and pretoken[i + 1] == token_id2:
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(pretoken[i])
                    i += 1
            new_pretoken_counts[tuple(new_word)] += c
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

    def train(self, input_text):
        self.get_initial_pretoken_counts(input_text)
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
    bpe = BPEVanilla(1000)
    with open(INPUT_FILE_PATH) as file:
        content = file.read()
    bpe.train(content)
    print(bpe.decode(bpe.encode("i love you")))


if __name__ == "__main__":
    main()
