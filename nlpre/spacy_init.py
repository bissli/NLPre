import os
import re

import spacy
from spacy.cli import download
from spacy.language import Language
from spacy.tokens import Doc


@Language.factory('dash_merger')
def dash_merger(nlp: Language, name: str):
    return DashMergerComponent(nlp)


class DashMergerComponent:
    """Add in the custom rule for joining dashed words as a single entity."""

    def __init__(self, nlp: Language):
        pass

    def __call__(self, doc: Doc) -> Doc:
        # This will be called on the Doc object in the pipeline
        expression = r'(?=\S*[-])([\S-]+)'
        with doc.retokenize() as retokenizer:
            for m in re.finditer(expression, doc.text):
                start, end = m.span()
                span = doc.char_span(start, end)
                retokenizer.merge(span)
        return doc


try:
    nlp = spacy.load('en_core_web_sm', exclude=['ner'])
except ModuleNotFoundError:
    # Hard code the model into NLPre
    download('en_core_web_sm-3.6.0', '--direct')
    nlp = spacy.load('en_core_web_sm', exclude=['ner'])

nlp.add_pipe('dash_merger', first=True)  # add it right after the tokenizer
