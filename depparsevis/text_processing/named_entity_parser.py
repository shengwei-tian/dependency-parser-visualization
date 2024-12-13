import nltk
import stanza
from tqdm import tqdm


class NamedEntityParser:
    def __init__(self):
        self.nlp = stanza.Pipeline(lang="nl", processors="tokenize,pos")

    def sent_parse(self, review):
        doc = self.nlp(review)
        return doc.sentences[0]

    def get_ner_tags(self, row):
        poses = row.pos
        for ixs, aspect in row.aspect2:
            try:
                for counter, ix in enumerate(ixs):
                    token, pos = poses[ix]
                    if counter == 0:
                        poses[ix] = (token, pos, f"B-{aspect.upper()}")
                    else:
                        poses[ix] = (token, pos, f"I-{aspect.upper()}")
            except TypeError:
                if len(poses[ixs]) == 2:
                    token, pos = poses[ixs]
                    poses[ixs] = (token, pos, f"B-{aspect.upper()}")

        for p in range(len(poses)):
            if len(poses[p]) == 2:
                token, pos = poses[p]
                poses[p] = (token, pos, "O")
        return poses

    def create_ner_dataset(self, df, path="./ner_dataset.txt"):
        with open(path, "a") as f:
            for _, sample in df.iterrows():
                lines = [" ".join(ner) + "\n" for ner in sample.ner]
                lines.append("\n")
                f.writelines(lines)

    def parse(self, df):
        tqdm.pandas(desc="Stanza Sentence Parsing")
        df["sent"] = df.text.progress_apply(lambda x: self.sent_parse(x))

        tqdm.pandas(desc="Extracting Tokens      ")
        df["tokens"] = df.sent.progress_apply(
            lambda x: [token.text for token in x.tokens]
        )

        tqdm.pandas(desc="Pos Tagging            ")
        df["pos"] = df.tokens.progress_apply(lambda x: nltk.pos_tag(x))

        tqdm.pandas(desc="NER Tagging            ")
        df["ner"] = df.progress_apply(lambda x: self.get_ner_tags(x), axis=1)
        df.drop(columns="sent", inplace=True)
        return df
