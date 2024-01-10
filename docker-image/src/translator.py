from transformers.tools import TranslationTool


def chunks(words, chunk_size):
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size])


class Translator(TranslationTool):
    def __call__(self, text, *args, chunk_size=400, **kwargs):
        res = []
        for chunk in chunks(text.split(" "), chunk_size):
            print("In chunk")
            res.append(super().__call__(chunk, *args, **kwargs))

        return " ".join(res)
