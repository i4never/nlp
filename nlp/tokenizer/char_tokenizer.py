import json
import logging
from collections import Counter
import tqdm
from typing import Dict, Optional, Mapping, List, Any

from nlp.tokenizer.tokenizer import Tokenizer, Token

logger = logging.getLogger(__name__)


@Tokenizer.register("char_tokenizer", exist_ok=True)
class CharTokenizer(Tokenizer):
    max_tokens: int = 512

    def __init__(self, token2id: Optional[Dict[str, int]] = None, **kwargs):
        if 'token2id_jsonl' in kwargs:
            with open(kwargs['token2id_jsonl']) as f:
                token2id = json.load(f)
        super(CharTokenizer, self).__init__(token2id=token2id, id2token={i: t for t, i in token2id.items()},
                                            **kwargs)

    def tokenize(self, text: str) -> List[Token]:
        return [Token(text=t, start=i, end=i + 1) for i, t in enumerate(text)]

    def token_to_indices(self, tokens: List[Token]) -> List[int]:
        return [self.token2id.get(token.text, self.token2id[self._token_unk]) for token in tokens]

    # @classmethod
    # def build(cls, data_loader: "DataLoader", path: str, max_vocab_size: int = 1e4, min_count: int = 10):
    #     print(f"从文件{path}使用{data_loader.__class__.__name__}构建vocab...")
    #     token_counter = Counter()
    #     for text in tqdm.tqdm(data_loader.iter(path)):
    #         for token in text:
    #             token_counter[token] += 1
    #
    #     token2id = {
    #         cls._token_pad: 0,
    #         cls._token_unk: 1,
    #         cls._token_sep: 2,
    #         cls._token_cls: 3,
    #         cls._token_mask: 4
    #     }
    #     for token, c in token_counter.most_common():
    #         if len(token2id) > max_vocab_size:
    #             break
    #         if c < min_count:
    #             break
    #         token2id[token] = len(token2id)
    #     print(f"从文件{path}使用{data_loader.__class__.__name__}构建vocab，size:{len(token2id)}")
    #     return cls(token2id=token2id)

    def _to_params(self) -> Dict[str, Any]:
        return self.__dict__

    def __repr_args__(self) -> 'ReprArgs':
        """
        hack token2id id2token的显示
        Returns:
        """
        args = list(super(CharTokenizer, self).__repr_args__())
        truncated = list()
        for k, v in args:
            if k == 'token2id' or k == 'id2token':
                v = str(dict(list(v.items())[:3])) + ' ... ' + str(len(v)) + ' tokens'
            truncated.append((k, v))

        return truncated


@Tokenizer.register("bert_char_tokenizer", exist_ok=True)
class BertCharTokenizer(CharTokenizer):
    def tokenize(self, text: str) -> List[Token]:
        tokens = super(BertCharTokenizer, self).tokenize(text)
        tokens = [Token(start=-1, end=-1, text=self._token_cls)] + tokens + \
                 [Token(start=len(text), end=len(text), text=self._token_sep)]
        return tokens


if __name__ == '__main__':
    from nlp.common.registrable import Registrable, import_all_modules_for_register
    from nlp.tokenizer.tokenizer import Tokenizer

    import_all_modules_for_register()

    for cls in Registrable.__subclasses__():
        print(cls.__name__, cls.list_available())

    tokenizer = Tokenizer().by_name("bert_char_tokenizer")(
        **{"token2id": {"a": 0, "b": 1, "<unk>": 2, "<cls>": 3, "<sep>": 4}, "max_tokens": 123})
    print(tokenizer.tokenize('这是一条测试语料'))

    print(tokenizer.token_to_indices(tokenizer.tokenize('这是一条测试语料')))

    # print("这是一条测试语料")
    # print(tokenizer.tokenize("这是一条测试语料"))
    # print(tokenizer.texts_to_id(["这是一条测试语料"]))
