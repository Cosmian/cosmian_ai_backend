# -*- coding: utf-8 -*-
from typing import List

import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

spacy.cli.download("en_core_web_sm")
spacy.cli.download("fr_core_news_sm")


class KeywordExtractor:
    def __init__(self, model_name: str, nb_keywords=10):
        # used for english text processing
        self.nlp_en_core_web_sm = spacy.load("en_core_web_sm")

        # used for FR text processing
        self.nlp_fr_core_news_sm = spacy.load("fr_core_news_sm")

        self.nb_keywords = nb_keywords
        self.kw_model = KeyBERT(model=SentenceTransformer(model_name))

    def __call__(self, text: str, src_lang: str) -> List[str]:
        keywords = self.extract_keywords(text)

        return self.post_process(keywords, src_lang)

    def extract_keywords(self, text: str) -> List[str]:
        keywords_scores = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            top_n=self.nb_keywords,
            stop_words=None,
        )

        return [keyword for keyword, _ in keywords_scores]

    def post_process(self, keywords: List[str], language: str) -> List[str]:
        # concatenate all keywords in a sentence
        keywords_text = " ".join(keywords)

        if language == "fr":
            doc = self.nlp_fr_core_news_sm(keywords_text)
        else:
            # english by default
            doc = self.nlp_en_core_web_sm(keywords_text)
        # lemmatize, remove duplicates and short words
        return list({token.lemma_ for token in doc if len(token.lemma_) >= 3})
