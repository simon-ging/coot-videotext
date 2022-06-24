"""
Text preprocessing functions.
"""
import re
from functools import partial
from typing import Callable, List, Optional

from nntrainer.typext import ConstantHolder


RE_WHITESPACES = re.compile(r"\s+")


class TextPreprocessing(ConstantHolder):
    """
    Enum for text preprocessing functions.
    """
    BERT_NEW = "bert_new"
    BERT_PAPER = "bert_paper"
    GPT2 = "gpt2"
    SIMPLE = "simple"
    NOTHING = "nothing"
    WITH_DOTS = "with_dots"


def get_text_preprocessor(func: str) -> Callable[[str], str]:
    """
    Given a string descriptor of the function, return the requested preprocessing function.

    Args:
        func: Function name.

    Returns:
        Text preprocessing function.
    """
    if func == TextPreprocessing.BERT_PAPER:
        # original implementation without dots and capitalization
        return partial(preprocess_paragraph, begin_paragraph_token="[CLS]", end_sentence_token="[SEP]",
                       remove_ending_dot=True, replace_inside_dots=True, capitalize=False)

    if func == TextPreprocessing.BERT_NEW:
        # new BERT implementation, no dots, with casing
        return partial(preprocess_paragraph, begin_paragraph_token="[CLS]", end_sentence_token="[SEP]",
                       remove_ending_dot=True, replace_inside_dots=True)
    if func == TextPreprocessing.GPT2:
        return partial(preprocess_paragraph, add_space_before_token=False)
    if func == TextPreprocessing.SIMPLE:
        return preprocess_paragraph
    if func == TextPreprocessing.NOTHING:
        return partial(preprocess_paragraph, capitalize=False)
    if func == TextPreprocessing.WITH_DOTS:
        return partial(preprocess_paragraph, remove_ending_dot=True, replace_inside_dots=True, capitalize=False)
    raise NotImplementedError(f"Text Processing '{func}' unknown")


def preprocess_paragraph(
        paragraph: List[str], begin_sentence_token: Optional[str] = None, end_sentence_token: Optional[str] = None,
        begin_paragraph_token: Optional[str] = None, end_paragraph_token: Optional[str] = None,
        add_space_before_token: bool = True,
        remove_ending_dot: bool = False, replace_inside_dots: bool = False, capitalize: bool = True) -> str:
    """
    Preprocess list of a paragraph into a single sentence.
    """
    new_paragraph = []

    space_before_token = " " if add_space_before_token else ""

    # define tokens between sentences
    between_sentence_token = None
    if end_sentence_token is not None or begin_sentence_token is not None:
        between_sentence_token = (f"{'' if end_sentence_token is None else f'{end_sentence_token} '}"
                                  f"{'' if begin_sentence_token is None else f'{begin_sentence_token}'}")

    for num_sentence, sentence in enumerate(paragraph):
        # strip and remove whitespaces
        sentence = RE_WHITESPACES.sub(" ", sentence).strip()
        assert len(sentence) > 0
        # remove last dot if requested, but keep multiple dots
        if remove_ending_dot:
            if sentence[-1] == "." and len(sentence) > 1 and sentence[-2] != ".":
                sentence = sentence[:-1]
        # add last dot if requested
        else:
            if sentence[-1] != ".":
                sentence += "."

        if capitalize:
            sentence = sentence.capitalize()
        sentence = sentence.strip()

        # dots inside sentences can happen in some datasets
        if capitalize:
            # capitalize after the dot
            find_pos = sentence.find(". ")
            if find_pos > -1:
                while True:
                    if find_pos > len(sentence):
                        break
                    find_pos += 1
                    if sentence[find_pos].isalnum():
                        sentence = sentence[:find_pos] + sentence[find_pos:].capitalize()
                        break

        if replace_inside_dots and between_sentence_token is not None:
            # add tokens and replace dot
            sentence = sentence.replace(". ", f"{'' if remove_ending_dot else '.'} {between_sentence_token} ")

        # start building new list of words
        new_words = []
        if begin_paragraph_token is not None and num_sentence == 0:
            new_words.append(begin_paragraph_token)
        if begin_sentence_token is not None:
            new_words.append(begin_sentence_token)

        # fill list of words, make sure there are no empty words or spaces
        words = sentence.split(" ")
        for word in words:
            word = word.strip()
            if word == "":
                continue
            new_words.append(f" {word.strip()}")

        # finish building new list of words
        if end_sentence_token is not None:
            new_words.append(f"{space_before_token}{end_sentence_token}")
        if end_paragraph_token is not None and num_sentence == len(paragraph) - 1:
            new_words.append(f"{space_before_token}{end_paragraph_token}")
        sentence = "".join(new_words).strip()
        new_paragraph.append(sentence)
    return new_paragraph
