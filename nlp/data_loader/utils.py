from typing import List
import jieba

import random


def full_to_half(text: str):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in text:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def create_masked_lm_predictions(tokens, vocab_words, masked_lm_prob=0.15,
                                 forbidden_tokens: List[str] = ['<cls>', '<sep>', '<pad>', '<mask>']):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in forbidden_tokens:
            continue
        cand_indexes.append([i])

    random.shuffle(cand_indexes)

    masked_tokens = list(tokens)

    num_to_predict = max(1, int(round(len(tokens) * masked_lm_prob)))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "<mask>"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(5, len(vocab_words) - 1)]

            masked_tokens[index] = masked_token

            masked_lms.append((index, tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])

    label_weights = [0. if i not in masked_lm_positions else 1. for i in range(len(masked_tokens))]
    return masked_tokens, label_weights, tokens


def create_whole_word_masked_lm_predictions(tokens, vocab_words, masked_lm_prob,
                                            forbidden_tokens: List[str] = ['<cls>', '<sep>', '<pad>', '<mask>']):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []

    offset = 0
    suffix_index = list()
    for word in jieba.cut(''.join(tokens)):
        suffix_index += [i for i in range(offset + 1, offset + len(word))]
        offset += len(word)

    for (i, token) in enumerate(tokens):
        if token in forbidden_tokens:
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if i in suffix_index:
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)

    masked_tokens = list(tokens)

    num_to_predict = max(1, int(round(len(tokens) * masked_lm_prob)))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "<mask>"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            masked_tokens[index] = masked_token

            masked_lms.append((index, tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])

    label_weights = [0. if i not in masked_lm_positions else 1. for i in range(len(masked_tokens))]
    return masked_tokens, label_weights, tokens
