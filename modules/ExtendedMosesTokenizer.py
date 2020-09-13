from sacremoses import MosesTokenizer, MosesDetokenizer
import re
from typing import List


MATCH_NUMBERS = r"(?<=\d)[,.](?=\d)", r" @\g<0>@ "
DETOKENIZE_NUMBERS = [(r" @\,@ ", r","),
                      (r" @\.@ ", r".")]


class ExtendedMosesTokenizer:
    """
    This class extends :class:`sacremoses.MosesTokenizer` and :class:`sacremoses.MosesDetokenizer`.

    The `ExtendedMosesTokenizer` implements the functionality to split large comma-separated numbers
    and floating point values. This is replacing commas with ' @,@ ' and dots with ' @.@ '.

    E.g. "23,000 people are 1.80m tall" -> "23 @,@ 000 people are 1 @.@ 80m tall".
    """

    def __init__(self):
        self.moses = MosesTokenizer()
        self.moses_d = MosesDetokenizer()

    def tokenize(self,
                 text: str,
                 aggressive_dash_splits=True,
                 escape=False,
                 return_str=False,
                 protected_patterns: List[str] = None) -> List[str]:
        """
        Tokenizes a string using :func:`sacremoses.MosesTokenizer.tokenize()` with additional
        tokenization of numbers.

        Examples::

            tokenizer = ExtendedMosesTokenizer()
            tokenizer.tokenize("I have (had) $5,000")
            # Expected result:
            # ['I', 'have', '(', 'had', ')', '$', '5', '@,@', '000']
        """

        if protected_patterns is None:
            protected_patterns = ['<eos>', '<cls>', '<unk>']
        mos = self.moses.tokenize(text, aggressive_dash_splits, return_str, escape, protected_patterns)
        mos_numbers = tokenize_numbers(mos)
        return mos_numbers

    def detokenize(self, text: str) -> str:
        numbers_d = detokenize_numbers(text)
        mos_d = self.moses_d.detokenize(numbers_d.split())
        return mos_d


def tokenize_numbers(text_array: List[str]) -> List[str]:
    tokenized = []
    for i in range(len(text_array)):
        replaced = re.sub(MATCH_NUMBERS[0], MATCH_NUMBERS[1], text_array[i]).split()
        tokenized.extend(replaced)

    return tokenized


def detokenize_numbers(text: str) -> str:
    for reg, sub in DETOKENIZE_NUMBERS:
        text = re.sub(reg, sub, text)
    return text