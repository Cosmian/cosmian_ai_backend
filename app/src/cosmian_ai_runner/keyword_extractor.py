# -*- coding: utf-8 -*-
from typing import List

import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

spacy.cli.download("en_core_web_sm")
spacy.cli.download("fr_core_news_sm")


class KeywordExtractor:
    def __init__(self, model_name: str, nb_keywords=10):

        # simple model to tokenize and lemmatize text
        self.nlp_pipeline = {
            "fr": spacy.load("fr_core_news_sm"),
            "en": spacy.load("en_core_web_sm"),
        }

        self.nb_keywords = nb_keywords
        # advanced model to find keywords from a text
        self.kw_model = KeyBERT(model=SentenceTransformer(model_name))

    def get_nlp_pipeline(self, language: str, default="en"):
        if language in self.nlp_pipeline:
            return self.nlp_pipeline[language]
        else:
            return self.nlp_pipeline[default]

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

        tokenized_keywords = self.get_nlp_pipeline(language)(keywords_text)

        # remove duplicates and short words
        return list(
            {token.text for token in tokenized_keywords if len(token.text) >= 3}
        )

    def lemmatize(self, keywords: List[str], language: str) -> List[str]:
        # concatenate all keywords in a sentence
        keywords_text = " ".join(keywords)

        tokenized_keywords = self.get_nlp_pipeline(language)(keywords_text)

        # remove duplicates and short words
        return list({token.lemma_ for token in tokenized_keywords})
