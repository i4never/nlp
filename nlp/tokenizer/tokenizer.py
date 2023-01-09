import logging
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Iterable, Mapping

from mesoorflow.common.registrable import Registrable

logger = logging.getLogger(__name__)


class Token(BaseModel):
    text: str
    start: int
    end: int


class Tokenizer(Registrable, BaseModel):
    """
    Tokenizer做什么:
        - tokenize / batch_tokenize: 文本分词
        - token_to_indices / batch_token_to_indices: token转文本id
        - post_process / batch_post_process: 后处理（例如Bert添加前后的<cls>与<sep>，限制文本长度等）
    Tokenizer不做什么:
        - 文本清洗（全、半脚转换）
    """
    token2id: Mapping[str, int] = dict()
    id2token: Mapping[int, str] = dict()
    _token_pad: str = '<pad>'
    _token_unk: str = '<unk>'
    _token_cls: str = '<cls>'
    _token_sep: str = '<sep>'
    _token_mask: str = '<mask>'
    _token_id_pad: int = 0

    @classmethod
    def build_from_corpus(cls, corpus: Iterable[str], **kwargs):
        raise NotImplementedError

    def texts_to_id(self, texts: List[str]) -> np.array:
        batch_tokens = self.batch_tokenize(texts)
        batch_token_ids = self.batch_token_to_indices(batch_tokens)
        batch_token_ids_padded = self.pad_to_longest(batch_token_ids)
        return batch_token_ids_padded

    def pad_to_longest(self, batch_token_ids: List[List[int]]) -> np.array:
        longest = max(len(token_ids) for token_ids in batch_token_ids)
        batch_token_ids_padded = np.ones((len(batch_token_ids), longest), dtype=np.int16) * self._token_id_pad
        for idx, token_id in enumerate(batch_token_ids):
            batch_token_ids_padded[idx, :len(token_id)] = token_id
        return batch_token_ids_padded

    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]

    def token_to_indices(self, tokens: List[Token]) -> List[int]:
        raise NotImplementedError

    def batch_token_to_indices(self, batch_tokens: List[List[Token]]) -> List[List[int]]:
        return [self.token_to_indices(tokens) for tokens in batch_tokens]

    def _to_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_params(cls, **kwargs):
        if 'name' not in kwargs:
            raise ValueError(f"初始化Tokenizer需要提供name")
        return cls.by_name(kwargs['name'])(**kwargs)

    @classmethod
    def load_from_file(cls, *args, **kwargs):
        raise NotImplementedError
