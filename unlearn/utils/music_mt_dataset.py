import copy

import numpy as np
import torch


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][: len(v)])
    return res


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return tuple([arg_number] * len(arg_structure))
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


class MusicMtDataset(torch.utils.data.Dataset):
    """
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
                        |
                        V
    [x1, x2, x3, x4, x5] [y1, y2, y3, y4, y5]
    """

    def __init__(
        self,
        src,
        src_sizes,
        tgt,
        tgt_sizes,
        src_vocab,
        tgt_vocab,
        left_pad_source=True,
        left_pad_target=False,
        max_source_positions=1024,
        max_target_positions=1024,
        shuffle=True,
        input_feeding=True,
        src_lang="",
        tgt_lang="",
    ):
        self.src = src  # MMap
        self.tgt = tgt  # MMap
        self.src_sizes = np.array(src_sizes)  # dataset.sizes()
        self.tgt_sizes = np.array(tgt_sizes)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        assert self.src_vocab.nspecial == self.tgt_vocab.nspecial

        self.sep_token = self.src_vocab.nspecial
        self.align_token = self.sep_token + 1

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]

        src_list = src_item.tolist()
        tgt_list = tgt_item.tolist()

        # For Source
        sep_positions = [i for i, x in enumerate(src_list) if x == self.sep_token]
        sep_positions.insert(0, -1)
        sentences = []
        for i in range(len(sep_positions) - 1):
            sentences.append(src_list[sep_positions[i] + 1: sep_positions[i + 1]])

        source = []
        source_sent_ids = []
        source_word_ids = []
        word_idx = 0
        for i, s in enumerate(sentences):
            for t in s:
                if t == self.align_token:
                    word_idx += 1
                else:
                    source.append(t)
                    source_word_ids.append(word_idx)
                    source_sent_ids.append(i)

            source.append(self.sep_token)
            source_sent_ids.append(i)
            source_word_ids.append(word_idx)
            word_idx += 1

        source.append(self.src_vocab.eos_index)
        source_sent_ids.append(-1)
        source_word_ids.append(word_idx)

        assert len(source) == len(source_sent_ids)
        assert len(source) == len(source_word_ids)

        # For Target
        sep_positions = [i for i, x in enumerate(tgt_list) if x == self.sep_token]
        sep_positions.insert(0, -1)
        sentences = []
        for i in range(len(sep_positions) - 1):
            sentences.append(tgt_list[sep_positions[i] + 1: sep_positions[i + 1]])

        target = []
        target_sent_ids = []
        target_word_ids = []
        word_idx = 0
        for i, s in enumerate(sentences):
            for (
                t
            ) in s:  # remove [align] which is 6, record word_id seqs and send_ids seq
                if t == self.align_token:
                    word_idx += 1
                else:
                    target.append(t)
                    target_word_ids.append(word_idx)
                    target_sent_ids.append(i)

            target.append(self.sep_token)  # Add sep token for every sentence
            target_sent_ids.append(i)
            target_word_ids.append(word_idx)
            word_idx += 1

        target.append(self.tgt_vocab.eos_index)  # append eos which 2
        target_sent_ids.append(-2)  # -2 for tgt non words
        target_word_ids.append(word_idx)  # eos token word align
        assert len(target) == len(target_sent_ids)
        assert len(target) == len(target_word_ids)

        return {
            "id": index,
            "source": torch.LongTensor(source),  # seq without [align]
            "target": torch.LongTensor(target),  # seq without [align]
            "source_sent_ids": torch.LongTensor(source_sent_ids),
            "target_sent_ids": torch.LongTensor(target_sent_ids),
            "source_word_ids": torch.LongTensor(source_word_ids),
            "target_word_ids": torch.LongTensor(target_word_ids),
        }

    def __len__(self):
        return len(self.src)

    def collate(
        self,
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True,
    ):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False):
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
                move_eos_to_beginning,
            )

        def merge_sentId(key, left_pad, pad_idx=pad_idx):
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
            )

        id = torch.LongTensor([s["id"] for s in samples])
        src_tokens = merge("source", left_pad=left_pad_source)
        # sort by descending source length
        src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        source_sent_ids = None
        source_sent_ids = merge_sentId(
            "source_sent_ids", left_pad=self.left_pad_target, pad_idx=-1
        )
        source_sent_ids = source_sent_ids.index_select(0, sort_order)
        source_word_ids = merge_sentId(
            "source_word_ids", left_pad=self.left_pad_target, pad_idx=-1
        )
        source_word_ids = source_word_ids.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        target_sent_ids = None

        if samples[0].get("target", None) is not None:
            target = merge("target", left_pad=left_pad_target)
            target = target.index_select(0, sort_order)
            ntokens = sum(len(s["target"]) for s in samples)

            target_sent_ids = merge_sentId(
                "target_sent_ids", left_pad=self.left_pad_target, pad_idx=-2
            )
            target_sent_ids = target_sent_ids.index_select(0, sort_order)
            target_word_ids = merge_sentId(
                "target_word_ids", left_pad=self.left_pad_target, pad_idx=-2
            )
            target_word_ids = target_word_ids.index_select(0, sort_order)

            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens = merge(
                    "target",
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        else:
            ntokens = sum(len(s["source"]) for s in samples)

        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "target": target,
            "word_ids": {
                "source_word_ids": source_word_ids,
                "target_word_ids": target_word_ids,
            },
        }
        if prev_output_tokens is not None:
            batch["net_input"]["prev_output_tokens"] = prev_output_tokens
        if source_sent_ids is not None:
            batch["net_input"]["source_sent_ids"] = source_sent_ids
        if target_sent_ids is not None:
            batch["net_input"]["target_sent_ids"] = target_sent_ids
        return batch

    def generate_dummy_batch(
        self, num_tokens, collate_fn, src_vocab, tgt_vocab, src_len=128, tgt_len=128
    ):
        """Return a dummy batch with a given number of tokens."""
        bsz = num_tokens // max(src_len, tgt_len)
        return collate_fn(
            [
                {
                    "id": i,
                    "source": src_vocab.dummy_sentence(src_len),
                    "target": tgt_vocab.dummy_sentence(tgt_len),
                    "output": tgt_vocab.dummy_sentence(tgt_len),
                }
                for i in range(bsz)
            ]
        )

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return self.collate(
            samples,
            pad_idx=self.src_vocab.pad(),
            eos_idx=self.src_vocab.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        return self.generate_dummy_batch(
            num_tokens, self.collater, self.src_vocab, self.tgt_vocab, src_len, tgt_len
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
