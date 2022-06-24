"""
Test text preprocessing (cleaning and adding special tokens).
"""
from pprint import pprint

from nntrainer import data_text


def test_text_preprocessing():
    input_paragraph = [
        "A man and a women introduce themselves to the camera. They start to talk to each other.",
        " As the conversation ensues another woman approaches the woman and takes away her water bottle.",
        " Afterwards the man starts to play the bongos and woman starts to dance.",
        " As the woman dances on a man on a bike passes by and starts to observe what's going on..."]
    pprint(input_paragraph)

    results = {
        data_text.TextPreprocessing.SIMPLE: [
            "A man and a women introduce themselves to the camera. They start to talk to each other.",
            "As the conversation ensues another woman approaches the woman and takes away her water bottle.",
            "Afterwards the man starts to play the bongos and woman starts to dance.",
            "As the woman dances on a man on a bike passes by and starts to observe what's going on..."],
        data_text.TextPreprocessing.BERT_PAPER: [
            "[CLS] A man and a women introduce themselves to the camera [SEP] They start to talk to each other [SEP]",
            "As the conversation ensues another woman approaches the woman and takes away her water bottle [SEP]",
            "Afterwards the man starts to play the bongos and woman starts to dance [SEP]",
            "As the woman dances on a man on a bike passes by and starts to observe what's going on... [SEP]"],
        data_text.TextPreprocessing.GPT2: [
            "A man and a women introduce themselves to the camera. They start to talk to each other.",
            "As the conversation ensues another woman approaches the woman and takes away her water bottle.",
            "Afterwards the man starts to play the bongos and woman starts to dance.",
            "As the woman dances on a man on a bike passes by and starts to observe what's going on..."]

    }

    for key, value in results.items():
        print("-" * 20, key)
        output_paragraph = data_text.get_text_preprocessor(key)(input_paragraph)
        assert value == output_paragraph, (
            f"Text preprocessing for {key} failed:\n\nOutput: {output_paragraph}\n\nTruth:  {value}")

    truth = [
        '[BOP][BOS] A man and a women introduce themselves to the camera. [EOS] [BOS] '
        'They start to talk to each other. [EOS]',
        '[BOS] As the conversation ensues another woman approaches the woman and takes away her water bottle. [EOS]',
        '[BOS] Afterwards the man starts to play the bongos and woman starts to dance. [EOS]',
        "[BOS] As the woman dances on a man on a bike passes by and starts to observe what's going on... [EOS] [EOP]"]
    output_paragraph = data_text.preprocess_paragraph(
        input_paragraph, begin_sentence_token="[BOS]", end_sentence_token="[EOS]",
        begin_paragraph_token="[BOP]", end_paragraph_token="[EOP]",
        remove_ending_dot=False, replace_inside_dots=True)
    assert output_paragraph == truth, f"Failed:\n\nOutput: {output_paragraph}\n\nTruth:  {truth}"


if __name__ == "__main__":
    test_text_preprocessing()
