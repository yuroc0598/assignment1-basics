import json
import os
import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from heapq import heapify, heappop, heappush
from itertools import pairwise
from multiprocessing import Pool
from typing import BinaryIO, TextIO

import regex as r


class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
        vocab_size: int | None = None,
        input_path: str | os.PathLike | None = None,
    ):
        special_tokens = list(special_tokens or [])
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special tokens must be unique")
        if any(not special for special in special_tokens):
            raise ValueError("special tokens must not be empty")

        self.PAT: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._pat_re = r.compile(self.PAT)
        self.special_tokens = special_tokens
        self._merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}

        self.merge_rules: list[tuple[bytes, bytes]] = []
        self._id_merge_rules: list[tuple[int, int, int]] = []

        if vocab is None:
            if vocab_size is None or input_path is None:
                raise ValueError("vocab_size and input_path are required for training")
            if merges is not None:
                raise ValueError("merges cannot be supplied when training")
            if vocab_size < 256 + len(special_tokens):
                raise ValueError("vocab_size must include all byte tokens and special tokens")
            self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
            self.vocab_size = vocab_size
            self.input_path = os.fspath(input_path)
            self._training_mode = True
        else:
            if vocab_size is not None or input_path is not None:
                raise ValueError("provide either vocab and merges, or vocab_size and input_path")
            if merges is None:
                raise ValueError("merges are required with vocab")
            self.vocab = dict(vocab)
            self.vocab_size = len(self.vocab)
            self.input_path: str | None = None
            self._training_mode = False

        self._add_special_tokens(special_tokens)
        self._validate_vocab()
        self.special_token_to_id = self._find_special_token_ids(special_tokens)
        if merges is not None:
            self._load_external_merges(merges)
        self._prepare_encoder()
        self.initial_vocab_size = len(self.vocab) - len(self.merge_rules)
        self.new_vocab_idx = max(self.vocab, default=-1) + 1

        self._special_re = (
            r.compile("|".join(r.escape(special) for special in sorted(special_tokens, key=len, reverse=True)))
            if special_tokens
            else None
        )
        self.pretoken_counts: defaultdict[tuple[int, ...], int] = defaultdict(
            int
        )  # {(token_idx1, token_idx2, token_idx3, ...): count)}

        self.pretoken_counts_with_index: defaultdict[int, tuple[int, ...]] = defaultdict(
            tuple
        )  # {pretoken_idx:(token_idx1, token_idx2, token_idx3, ..., count)}
        if special_tokens:
            if "<|endoftext|>" in special_tokens:
                self.end_of_doc = "<|endoftext|>"
            else:
                self.end_of_doc = special_tokens[0]
        else:
            self.end_of_doc = "<|endoftext|>"
        self.pair_buckets = {}  # {count: set[(token_id_a, token_id_b, byte_a, byte_b)]}
        self.pair_count = {}  # {(p1, p2): [c, set(pretoken_idx)]}
        self.max_pair_count = 0

    def _add_special_tokens(self, special_tokens: list[str]) -> None:
        existing_tokens = set(self.vocab.values())
        next_token_id = max(self.vocab, default=-1) + 1
        for special in special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes not in existing_tokens:
                self.vocab[next_token_id] = special_bytes
                existing_tokens.add(special_bytes)
                next_token_id += 1

    def _validate_vocab(self) -> None:
        if not all(
            isinstance(token_id, int) and isinstance(token_bytes, bytes) for token_id, token_bytes in self.vocab.items()
        ):
            raise TypeError("vocab must map integer token IDs to bytes")
        self._ids_by_bytes: defaultdict[bytes, list[int]] = defaultdict(list)
        for token_id, token_bytes in self.vocab.items():
            self._ids_by_bytes[token_bytes].append(token_id)
        for token_ids in self._ids_by_bytes.values():
            token_ids.sort()

        missing_bytes = [byte for byte in range(256) if bytes([byte]) not in self._ids_by_bytes]
        if missing_bytes:
            raise ValueError("vocab must contain a token for every byte value")

    def _find_special_token_ids(self, special_tokens: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        used_ids: set[int] = set()
        for special in special_tokens:
            special_bytes = special.encode("utf-8")
            candidates = [token_id for token_id in self._ids_by_bytes[special_bytes] if token_id not in used_ids]
            if not candidates:
                raise ValueError(f"special token {special!r} is missing from vocab")
            token_id = candidates[0]
            result[special] = token_id
            used_ids.add(token_id)
        return result

    def _base_byte_token_ids(self) -> dict[int, int]:
        special_ids = set(self.special_token_to_id.values())
        result: dict[int, int] = {}
        for byte in range(256):
            token_bytes = bytes([byte])
            candidates = [token_id for token_id in self._ids_by_bytes[token_bytes] if token_id not in special_ids]
            if not candidates:
                raise ValueError(f"vocab is missing ordinary byte token {byte}")
            result[byte] = byte if byte in candidates else candidates[0]
        return result

    def _load_external_merges(self, merges: list[tuple[bytes, bytes]]) -> None:
        """Convert the public byte-pair merge format to internal token-ID rules."""
        special_ids = set(self.special_token_to_id.values())
        available_ids = set(self._base_byte_token_ids().values()) | special_ids

        ordinary_token_to_id = {
            token_bytes: next(
                (token_id for token_id in token_ids if token_id not in special_ids),
                None,
            )
            for token_bytes, token_ids in self._ids_by_bytes.items()
        }

        def available_id_for(token_bytes: bytes) -> int:
            token_id = ordinary_token_to_id.get(token_bytes)
            if token_id is None or token_id not in available_ids:
                raise ValueError(f"merge references token bytes not yet available: {token_bytes!r}")
            return token_id

        for left_bytes, right_bytes in merges:
            if not isinstance(left_bytes, bytes) or not isinstance(right_bytes, bytes):
                raise TypeError("each merge must be a pair of bytes")
            left_id = available_id_for(left_bytes)
            right_id = available_id_for(right_bytes)
            merged_bytes = left_bytes + right_bytes
            output_id = ordinary_token_to_id.get(merged_bytes)
            if output_id is None:
                raise ValueError(f"vocab has no output token for merge {(left_bytes, right_bytes)!r}")
            self._id_merge_rules.append((left_id, right_id, output_id))
            self.merge_rules.append((left_bytes, right_bytes))
            available_ids.add(output_id)

    def _prepare_encoder(self) -> None:
        self._validate_vocab()
        self._byte_token_ids = self._base_byte_token_ids()
        special_ids = set(self.special_token_to_id.values())
        self._ordinary_token_to_id: dict[bytes, int] = {}
        for token_bytes, token_ids in self._ids_by_bytes.items():
            token_id = next(
                (token_id for token_id in token_ids if token_id not in special_ids),
                None,
            )
            if token_id is not None:
                self._ordinary_token_to_id[token_bytes] = token_id
        self._merge_ranks = {
            (left_id, right_id): (rank, output_id)
            for rank, (left_id, right_id, output_id) in enumerate(self._id_merge_rules)
        }

    def _reset_training_state(self) -> None:
        if not self._training_mode:
            raise RuntimeError("a BPE initialized from a vocabulary cannot be retrained")
        self.vocab = {i: bytes([i]) for i in range(256)}
        for special, token_id in self.special_token_to_id.items():
            self.vocab[token_id] = special.encode("utf-8")
        self._id_merge_rules.clear()
        self.merge_rules.clear()
        self._prepare_encoder()
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
            chunks = self._special_re.split(text) if self._special_re else (text,)
            for chunk in chunks:
                for match in self._pat_re.finditer(chunk):
                    pretoken = tuple(match.group().encode("utf-8"))
                    local_pretoken_counts[pretoken] += 1

        return local_pretoken_counts

    def get_initial_pretoken_counts_from_file(self, input_path: str) -> defaultdict[tuple[int, ...], int]:
        if not input_path:
            raise ValueError("input file/path is not provided")

        self._reset_training_state()

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
        self._id_merge_rules.append((token_id_a, token_id_b, new_token_id))
        self.merge_rules.append((byte_a, byte_b))
        # update self.pretoken_counts, only update the pretokens that the pair (a, b) occurred
        # 1. remove occurrance of (a, b) from pretoken_counts, and update the bucket and heap
        # 2. if there is item before (a, b), say (x, a, b), then reduce the counts of (x, a), add the count of (x, ab)
        # 3. if there is item after, say (a, b, y), then reduce the count of (b, y), and increase the count of (ab, y)
        pretoken_ids_occurred = set(self.pair_count[(token_id_a, token_id_b)][1])
        for pretoken_idx in pretoken_ids_occurred:
            tokens = self.pretoken_counts_with_index[pretoken_idx][:-1]
            pretoken_count = self.pretoken_counts_with_index[pretoken_idx][-1]
            new_tokens = self._merge_pair_everywhere(tokens, token_id_a, token_id_b, new_token_id)
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

    def _merge_pair_everywhere(self, input_token_id_list, p_a, p_b, c):
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

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        if not self._training_mode:
            raise RuntimeError("a BPE initialized from a vocabulary cannot be retrained")
        self.get_initial_pretoken_counts_from_file(self.input_path)

        while len(self.vocab) < self.vocab_size:
            self.get_max_pair_and_merge()
        self._prepare_encoder()
        return self.vocab, self.merge_rules

    def _encode_pretoken(self, pretoken: bytes) -> tuple[int, ...]:
        token_id = self._ordinary_token_to_id.get(pretoken)
        if token_id is not None:
            return (token_id,)

        token_ids = [self._byte_token_ids[byte] for byte in pretoken]
        if len(token_ids) < 2 or not self._merge_ranks:
            return tuple(token_ids)
        if len(token_ids) < 100:
            return self._merge_short_pretoken(token_ids)
        return self._merge_large_pretoken(token_ids)

    def _merge_short_pretoken(self, token_ids: list[int]) -> tuple[int, ...]:
        """Use a cache-friendly linear scan for normal, short pretokens."""
        while len(token_ids) > 1:
            best_rank = len(self._id_merge_rules)
            best_index = -1
            best_output_id = -1
            for index, (left_id, right_id) in enumerate(pairwise(token_ids)):
                merge = self._merge_ranks.get((left_id, right_id))
                if merge is not None and merge[0] < best_rank:
                    best_rank, best_output_id = merge
                    best_index = index

            if best_index == -1:
                break
            token_ids[best_index] = best_output_id
            del token_ids[best_index + 1]
        return tuple(token_ids)

    def _merge_large_pretoken(self, token_ids: list[int]) -> tuple[int, ...]:
        """Use a heap and linked indices for unusually large pretokens."""
        token_count = len(token_ids)
        previous = [index - 1 for index in range(token_count)]
        following = [index + 1 for index in range(token_count)]
        following[-1] = -1
        alive = bytearray(b"\x01") * token_count

        heap: list[tuple[int, int, int, int]] = []
        for left_index in range(token_count - 1):
            right_index = left_index + 1
            merge = self._merge_ranks.get((token_ids[left_index], token_ids[right_index]))
            if merge is not None:
                rank, output_id = merge
                heap.append((rank, left_index, right_index, output_id))
        heapify(heap)

        def add_candidate(left_index: int, right_index: int) -> None:
            if left_index == -1 or right_index == -1:
                return
            merge = self._merge_ranks.get((token_ids[left_index], token_ids[right_index]))
            if merge is not None:
                rank, output_id = merge
                heappush(heap, (rank, left_index, right_index, output_id))

        while heap:
            rank, left_index, right_index, output_id = heappop(heap)
            if not alive[left_index] or not alive[right_index] or following[left_index] != right_index:
                continue
            if self._merge_ranks.get((token_ids[left_index], token_ids[right_index])) != (rank, output_id):
                continue

            token_ids[left_index] = output_id
            alive[right_index] = 0
            next_index = following[right_index]
            following[left_index] = next_index
            if next_index != -1:
                previous[next_index] = left_index

            add_candidate(previous[left_index], left_index)
            add_candidate(left_index, next_index)

        encoded: list[int] = []
        index = 0
        while index != -1:
            encoded.append(token_ids[index])
            index = following[index]
        return tuple(encoded)

    def _encode_ordinary_range(self, text: str, start: int, end: int) -> Iterator[int]:
        for match in self._pat_re.finditer(text, start, end):
            yield from self._encode_pretoken(match.group().encode("utf-8"))

    def encode_iter(self, text: str) -> Iterator[int]:
        """Yield token IDs without materializing the entire encoded result."""
        if self._special_re is None:
            yield from self._encode_ordinary_range(text, 0, len(text))
            return

        position = 0
        for match in self._special_re.finditer(text):
            yield from self._encode_ordinary_range(text, position, match.start())
            yield self.special_token_to_id[match.group()]
            position = match.end()
        yield from self._encode_ordinary_range(text, position, len(text))

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        """Encode independent text chunks lazily, such as lines from a large file."""
        for text in texts:
            yield from self.encode_iter(text)

    def encode_file(
        self,
        input_path: str | os.PathLike,
        output_path: str | os.PathLike,
        dtype: str = "uint32",
        buffer_size: int = 65_536,
        num_workers: int = 1,
        batch_size: int = 128,
        document_separator: str | None = None,
    ) -> int:
        """Stream a UTF-8 text file into a new little-endian binary token file.

        The output can be read with ``np.memmap(output_path, dtype="<u4")`` for
        uint32 or ``dtype="<u2"`` for uint16. ``num_workers > 1`` encodes
        batches in worker processes while preserving input order. When supplied,
        ``document_separator`` must be a configured special token and is used as
        a tokenization-safe record boundary. Otherwise, records are input lines.
        Returns the number of token IDs. Existing output files are not overwritten.
        """
        typecodes = {"uint16": ("H", 2**16 - 1), "uint32": ("I", 2**32 - 1)}
        if dtype not in typecodes:
            raise ValueError("dtype must be 'uint16' or 'uint32'")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if document_separator is not None and document_separator not in self.special_token_to_id:
            raise ValueError("document_separator must be a configured special token")

        typecode, max_token_id = typecodes[dtype]
        if max(self.vocab, default=0) > max_token_id:
            raise ValueError(f"token IDs do not fit in {dtype}")

        token_count = 0
        token_buffer = array(typecode)
        expected_item_size = 2 if dtype == "uint16" else 4
        if token_buffer.itemsize != expected_item_size:
            raise RuntimeError(f"platform does not provide a {dtype} array type")

        def record_batches(input_file: TextIO) -> Iterator[list[str]]:
            batch: list[str] = []
            for record in self._iter_file_records(input_file, document_separator):
                batch.append(record)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def append_tokens(token_ids: Iterable[int], output_file: BinaryIO) -> None:
            nonlocal token_count, token_buffer
            for token_id in token_ids:
                token_buffer.append(token_id)
                token_count += 1
                if len(token_buffer) >= buffer_size:
                    if sys.byteorder != "little":
                        token_buffer.byteswap()
                    token_buffer.tofile(output_file)
                    token_buffer = array(typecode)

        with (
            open(input_path, encoding="utf-8", newline="") as input_file,
            open(output_path, "xb") as output_file,
        ):
            batches = record_batches(input_file)
            if num_workers == 1:
                for batch in batches:
                    append_tokens(self.encode_iterable(batch), output_file)
            else:
                with Pool(
                    num_workers,
                    initializer=_initialize_encode_worker,
                    initargs=(self.vocab, self.merge_rules, self.special_tokens),
                ) as pool:
                    for token_ids in pool.imap(_encode_record_batch, batches):
                        append_tokens(token_ids, output_file)

            if token_buffer:
                if sys.byteorder != "little":
                    token_buffer.byteswap()
                token_buffer.tofile(output_file)

        return token_count

    @staticmethod
    def _iter_file_records(
        input_file: TextIO,
        document_separator: str | None,
        read_size: int = 1 << 20,
    ) -> Iterator[str]:
        if document_separator is None:
            yield from input_file
            return

        separator_length = len(document_separator)
        pending_parts: list[str] = []
        overlap = ""
        while block := input_file.read(read_size):
            data = overlap + block
            position = 0
            while (found_at := data.find(document_separator, position)) != -1:
                record_end = found_at + separator_length
                pending_parts.append(data[position:record_end])
                yield "".join(pending_parts)
                pending_parts.clear()
                position = record_end

            overlap_length = min(separator_length - 1, len(data) - position)
            stable_end = len(data) - overlap_length
            pending_parts.append(data[position:stable_end])
            overlap = data[stable_end:]

        pending_parts.append(overlap)
        if any(pending_parts):
            yield "".join(pending_parts)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iter(text))

    def decode(self, token_ids) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        return token_bytes.decode("utf-8", errors="replace")

    def save(self, path: str | os.PathLike, overwrite: bool = False) -> None:
        """Save the vocabulary, ordered merges, and special tokens as JSON."""
        payload = {
            "format": "cs336-bpe",
            "version": 1,
            "vocab": [[token_id, token_bytes.hex()] for token_id, token_bytes in sorted(self.vocab.items())],
            "merges": [[left_bytes.hex(), right_bytes.hex()] for left_bytes, right_bytes in self.merge_rules],
            "special_tokens": self.special_tokens,
        }
        mode = "w" if overwrite else "x"
        with open(path, mode, encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | os.PathLike):
        """Load a tokenizer written by :meth:`save`."""
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("format") != "cs336-bpe" or payload.get("version") != 1:
            raise ValueError("unsupported tokenizer file format")

        try:
            vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in payload["vocab"]}
            merges = [(bytes.fromhex(left_hex), bytes.fromhex(right_hex)) for left_hex, right_hex in payload["merges"]]
            special_tokens = list(payload["special_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid tokenizer file") from error

        return cls(vocab, merges, special_tokens)


_ENCODE_WORKER: BPE | None = None


def _initialize_encode_worker(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    global _ENCODE_WORKER
    _ENCODE_WORKER = BPE(vocab, merges, special_tokens)


def _encode_record_batch(records: list[str]) -> list[int]:
    if _ENCODE_WORKER is None:
        raise RuntimeError("encode worker was not initialized")
    return list(_ENCODE_WORKER.encode_iterable(records))
