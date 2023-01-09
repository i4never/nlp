import os

import json
from typing import List, Dict

import numpy as np
from pydantic import BaseModel

from nlp.settings import logger
from nlp.common.registrable import Registrable
from nlp.data_loader.utils import full_to_half, create_whole_word_masked_lm_predictions
from nlp.tokenizer.tokenizer import Tokenizer
import pandas as pd
import tensorflow as tf


class DataLoader(Registrable, BaseModel):
    full_to_half: bool = True
    lowercase: bool = False

    def build_input_grt(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def types(self) -> Dict:
        raise NotImplementedError

    @property
    def shapes(self) -> Dict:
        raise NotImplementedError

    @property
    def padded_shapes(self) -> Dict:
        raise NotImplementedError

    def preprocess(self, text: str) -> str:
        if self.full_to_half:
            text = full_to_half(text)
        if self.lowercase:
            text = text.lower()
        return text

    @classmethod
    def from_params(cls, **kwargs):
        if 'name' not in kwargs:
            raise ValueError(f"初始化DataLoader需要提供name")
        return cls.by_name(kwargs['name'])(**kwargs)


@DataLoader.register("ner_jsonl_bio")
class NerJsonlBIODataLoader(DataLoader):
    entities: List[str]

    def build_input_grt(self, path: str, tokenizer: Tokenizer):
        entity2id = {e: i for i, e in enumerate(self.entities)}

        def input_generator():
            with open(path) as f:
                for l in f:
                    record = json.loads(l)
                    text = record['text']
                    spans = record['labels']

                    tokens = tokenizer.tokenize(text)
                    b_index_to_labels = {start: entity for start, end, entity in spans}
                    i_index_to_labels = {i: entity for start, end, entity in spans for i in range(start + 1, end)}

                    target_seq = list()
                    for token in tokens:
                        target_id = 0
                        for idx in range(token.start, token.end):
                            if idx in b_index_to_labels:
                                target_id = 2 * entity2id[b_index_to_labels[idx]] + 1
                                break
                            if idx in i_index_to_labels:
                                target_id = 2 * entity2id[i_index_to_labels[idx]] + 2
                        target_seq.append(target_id)
                    assert len(tokens) == len(target_seq)
                    yield {
                        "inputs": {
                            "batch_token_ids": np.array(tokenizer.token_to_indices(tokens)).reshape(-1)
                        },
                        "outputs": {
                            "batch_token_labels": np.array(target_seq).reshape(-1),
                            "batch_sequence_length": np.array(len(tokens)).reshape(1),
                        }
                    }

        return input_generator

    @property
    def inputs(self):
        return {
            "batch_token_ids": tf.placeholder(tf.int32, [None, None], name="batch_token_ids")
        }

    @property
    def types(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": tf.int32
            },
            "outputs": {
                "batch_token_labels": tf.int32,
                "batch_sequence_length": tf.int32
            }
        }

    @property
    def shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [None]
            },
            "outputs": {
                "batch_token_labels": [None],
                "batch_sequence_length": [1]
            }
        }

    @property
    def padded_shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [-1]
            },
            "outputs": {
                "batch_token_labels": [-1],
                "batch_sequence_length": [1]
            }
        }


@DataLoader.register("ner_jsonl_global_pointer")
class NerJsonlGlobalPointerDataLoader(DataLoader):
    entities: List[str]

    def build_input_grt(self, path: str, tokenizer: Tokenizer):
        entity2id = {e: i for i, e in enumerate(self.entities)}

        def input_generator():
            with open(path) as f:
                for l in f:
                    record = json.loads(l)
                    text = record['text']
                    spans = record['labels']

                    tokens = tokenizer.tokenize(text)
                    char_idx_to_token_idx = {char_idx: token_idx for token_idx, token in enumerate(tokens) for char_idx
                                             in range(token.start, token.end)}
                    pointer = np.zeros(shape=(len(tokens), len(tokens)))

                    for char_start, char_end, entity in spans:
                        token_start, token_end = char_idx_to_token_idx[char_start], char_idx_to_token_idx[char_end - 1]
                        pointer[token_start, token_end] = entity2id[entity]

                    yield {
                        "inputs": {
                            "batch_token_ids": np.array(tokenizer.token_to_indices(tokens)).reshape(-1)
                        },
                        "outputs": {
                            "batch_global_pointer": pointer
                        }
                    }

        return input_generator

    @property
    def inputs(self):
        return {
            "batch_token_ids": tf.placeholder(tf.int32, [None, None], name="batch_token_ids")
        }

    @property
    def types(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": tf.int32
            },
            "outputs": {
                "batch_global_pointer": tf.int32
            }
        }

    @property
    def shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [None]
            },
            "outputs": {
                "batch_global_pointer": [None, None]
            }
        }

    @property
    def padded_shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [-1]
            },
            "outputs": {
                "batch_global_pointer": [-1, -1],
            }
        }


@DataLoader.register("rich_document_mlm", exist_ok=True)
class RichDocumentMLMDataLoader(DataLoader):
    """
    输入由以下几部分组成：
        - token id
        - top y (normalize to 1000)
        - bottom y (normalize to 1000)
        - left x (normalize to 1000)
        - right x (normalize to 1000)
        - page_id (max to 8)
    特别的
        - vocab中没有控制字符
        - 大于8的页会被丢弃
    """
    max_width: int = 1000
    max_height: int = 1000
    max_page: int = 8
    max_length: int = 8000

    def build_input_grt(self, path: str, tokenizer: Tokenizer, mask_rate: float = 0.15):
        paths = os.listdir(path)
        import random

        random.shuffle(paths)

        def input_generator():
            for p in paths:
                tokens = list()
                x_left_ids = list()
                x_right_ids = list()
                y_top_ids = list()
                y_bottom_ids = list()
                page_ids = list()
                with open(f"{path}/{p}") as f:
                    try:
                        # record = json.load(f)
                        df = pd.read_csv(f)
                        df = df[1:]
                    except Exception as e:
                        logger.warning(f"{path}/{p} load error")
                        continue
                    for _, r in df.iterrows():
                        if r.page_idx >= self.max_page:
                            logger.debug(f"文件{path}/{p}丢弃大于{self.max_page}的页")
                            break

                        try:
                            if pd.isna(r.text) or r.text.strip() == '':
                                # 丢弃所有空格
                                continue
                        except Exception as e:
                            print(r.text, e)
                            raise e
                        if len(tokens) >= self.max_length:
                            # logger.warning(f"截断")
                            break
                        if r.x < 0 or r.x + r.width > r.page_width:
                            continue
                        if r.y - r.height < 0 or r.y > r.page_height:
                            continue

                        tokens.append(
                            r.text if r.text in tokenizer.token2id else tokenizer._token_unk)

                        x_left_ids.append(int(self.max_width * r.x / r.page_width))
                        x_right_ids.append(int(self.max_width * (r.x + r.width) / r.page_width))
                        y_top_ids.append(int(self.max_height * (r.y - r.height) / r.page_height))
                        y_bottom_ids.append(int(self.max_height * r.y / r.page_height))
                        page_ids.append(r.page_idx)

                if len(tokens) > 10000 or len(tokens) < 100:
                    continue

                # dynamic mask
                input_tokens, weights, output_tokens = \
                    create_whole_word_masked_lm_predictions(tokens, list(tokenizer.token2id.items()), mask_rate)
                input_token_ids = [tokenizer.token2id.get(tk, tokenizer.token2id[tokenizer._token_unk]) for tk in
                                   input_tokens]
                output_token_ids = [tokenizer.token2id.get(tk, tokenizer.token2id[tokenizer._token_unk]) for tk in
                                    output_tokens]

                # 添加cls sep的添加，根据layoutlm v1的图示，pos embedding分别填充0,0,maxW,maxH
                input_token_ids = [tokenizer.token2id[tokenizer._token_cls]] + input_token_ids + \
                                  [tokenizer.token2id[tokenizer._token_sep]]
                output_token_ids = [tokenizer.token2id[tokenizer._token_cls]] + output_token_ids + \
                                   [tokenizer.token2id[tokenizer._token_sep]]
                x_left_ids = [0] + x_left_ids + [self.max_width - 1]
                x_right_ids = [0] + x_right_ids + [self.max_width - 1]
                y_top_ids = [0] + y_top_ids + [self.max_height - 1]
                y_bottom_ids = [0] + y_bottom_ids + [self.max_height - 1]
                weights = [0.] + weights + [0.]
                page_ids = [page_ids[0]] + page_ids + [page_ids[-1]]

                assert len(input_token_ids) == len(x_right_ids) == len(x_left_ids) == len(y_top_ids) == len(
                    y_bottom_ids) == len(output_token_ids) == len(page_ids) == len(weights)

                yield {
                    "inputs": {
                        "batch_token_ids": np.array(input_token_ids).reshape(-1),
                        "batch_x_left_ids": np.array(x_left_ids).reshape(-1),
                        "batch_x_right_ids": np.array(x_right_ids).reshape(-1),
                        "batch_y_top_ids": np.array(y_top_ids).reshape(-1),
                        "batch_y_bottom_ids": np.array(y_bottom_ids).reshape(-1),
                        "batch_page_ids": np.array(page_ids).reshape(-1)
                    },
                    "outputs": {
                        "batch_label_ids": np.array(output_token_ids).reshape(-1)
                    },
                    "weights": {
                        "batch_label_weights": np.array(weights).reshape(-1)
                    }
                }

        return input_generator

    @property
    def types(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": tf.int32,
                "batch_x_left_ids": tf.int32,
                "batch_x_right_ids": tf.int32,
                "batch_y_top_ids": tf.int32,
                "batch_y_bottom_ids": tf.int32,
                "batch_page_ids": tf.int32
            },
            "outputs": {
                "batch_label_ids": tf.int32,
            },
            "weights": {
                "batch_label_weights": tf.float32
            }
        }

    @property
    def shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [None],
                "batch_x_left_ids": [None],
                "batch_x_right_ids": [None],
                "batch_y_top_ids": [None],
                "batch_y_bottom_ids": [None],
                "batch_page_ids": [None]
            },
            "outputs": {
                "batch_label_ids": [None]
            },
            "weights": {
                "batch_label_weights": [None]
            }
        }

    @property
    def padded_shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [-1],
                "batch_x_left_ids": [-1],
                "batch_x_right_ids": [-1],
                "batch_y_top_ids": [-1],
                "batch_y_bottom_ids": [-1],
                "batch_page_ids": [-1]
            },
            "outputs": {
                "batch_label_ids": [-1]
            },
            "weights": {
                "batch_label_weights": [-1]
            }
        }


@DataLoader.register("pairwise_cls_jsonl", exist_ok=True)
class PairwiseClsJsonlDataLoader(DataLoader):
    def build_input_grt(self, path: str, *args, **kwargs):
        def input_generator():
            with open(path) as f:
                for l in f:
                    r = json.loads(l)
                    yield {
                        "inputs": {
                            "batch_token_ids": np.array(r["job_token_ids"][:-1] + r["resume_token_ids"][1:]).reshape(-1)
                        },
                        "outputs": {
                            "batch_label_ids": np.array([r["cls_label"]]).reshape([1])
                        }
                    }

        return input_generator

    @property
    def types(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": tf.int32
            },
            "outputs": {
                "batch_label_ids": tf.int32
            }
        }

    @property
    def shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [None]
            },
            "outputs": {
                "batch_label_ids": [1]
            }
        }

    @property
    def padded_shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [-1]
            },
            "outputs": {
                "batch_label_ids": [-1]
            }
        }


@DataLoader.register("pairwise_bio_jsonl", exist_ok=True)
class PairwiseBIOJsonlDataLoader(DataLoader):
    def build_input_grt(self, path: str, *args, **kwargs):
        def input_generator():
            with open(path) as f:
                for l in f:
                    r = json.loads(l)
                    token_labels = [0 for _ in range(len(r["job_token_ids"]) + len(r["resume_token_ids"]))]
                    offset = len(r["job_token_ids"])

                    for st, ed in r['reason_spans']:
                        token_labels[offset + st] = 1
                        for i in range(offset + st + 1, offset + ed):
                            token_labels[i] = 2

                    yield {
                        "inputs": {
                            "batch_token_ids": np.array(r["job_token_ids"] + r["resume_token_ids"]).reshape(-1)
                        },
                        "outputs": {
                            "batch_token_labels": np.array(token_labels).reshape(-1)
                        }
                    }

        return input_generator

    @property
    def types(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": tf.int32
            },
            "outputs": {
                "batch_token_labels": tf.int32
            }
        }

    @property
    def shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [None]
            },
            "outputs": {
                "batch_token_labels": [None]
            }
        }

    @property
    def padded_shapes(self) -> Dict:
        return {
            "inputs": {
                "batch_token_ids": [-1]
            },
            "outputs": {
                "batch_token_labels": [-1]
            }
        }


if __name__ == '__main__':
    data_loader = DataLoader.from_params(**{
        "name": "pairwise_bio_jsonl"
    })

    tf.enable_eager_execution()

    dataset = tf.data.Dataset.from_generator(
        data_loader.build_input_grt(
            '/Users/i4never/Documents/workspace/model-finetune/answer_20221220_0_train_1.json'),
        data_loader.types,
        data_loader.shapes
    )

    dataset = dataset.repeat(1000)
    dataset = dataset.padded_batch(4, data_loader.padded_shapes)
    dataset = dataset.prefetch(32)
    for batch in dataset:
        print(batch)
        input()
