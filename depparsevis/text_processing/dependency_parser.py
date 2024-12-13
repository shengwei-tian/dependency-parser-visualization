import pickle
import re

import numpy as np
import stanza
from tqdm import tqdm


class DependencyParser:
    def __init__(self, lang="nl", processors="tokenize,pos,lemma,depparse"):
        self.nlp = stanza.Pipeline(lang=lang, processors=processors)

    def sent_parse(self, review):
        doc = self.nlp(review)
        return doc.sentences[0]

    @staticmethod
    def get_tokens_norm(sent):
        return [word.text.lower() for word in sent.words]

    def create_parse_tree(self, sent):
        root = [word for word in sent.words if word.head == 0][0]
        tree_norm = [("ROOT", root.head, root.id)]
        for word in sent.words:
            if word.head == 0:
                continue
            elif word.head > 0:
                tree_norm.append((word.deprel, word.head, word.id))
        return tree_norm

    def create_clean_tree_norm(self, tree_norm):
        return [(head - 1, word - 1) for _, head, word in tree_norm[1:]]

    @staticmethod
    def get_root_norm(tree):
        return tree[0][-1] - 1

    def get_token_ids(self, sent, aspect):
        closest_to_begin, closest_to_end = None, None

        begin_diff = len(sent.text)
        end_diff = len(sent.text)
        for token in sent.tokens:
            diff = abs(aspect[0] - token.start_char)
            if diff < begin_diff:
                begin_diff = diff
                closest_to_begin = token

            e_diff = abs(aspect[1] - token.end_char)
            if e_diff < end_diff:
                end_diff = e_diff
                closest_to_end = token

        if (closest_to_begin is None) or (closest_to_end is None):
            return None

        if closest_to_begin == closest_to_end:
            return (closest_to_begin.id[0] - 1, aspect[2])
        else:
            return (
                list(np.arange(closest_to_begin.id[0] - 1, closest_to_end.id[0])),
                aspect[2],
            )

    def parse(
        self,
        df,
        save=False,
        save_path="/Volumes/LUMC/Data/annotated_data/annotated_all.pkl",
    ):
        if "sent" not in df.columns:
            tqdm.pandas(desc="Stanza Sentence Parsing")
            df["sent"] = df.text.progress_apply(lambda x: self.sent_parse(x))

        tqdm.pandas(desc="Aspect Parsing         ")
        df["aspect2"] = df.progress_apply(
            lambda x: [self.get_token_ids(x["sent"], aspect) for aspect in x["aspect"]],
            axis=1,
        )

        tqdm.pandas(desc="Dependency Tree Parsing")
        df["dependency_tree_norm"] = df.progress_apply(
            lambda x: self.create_parse_tree(x["sent"]), axis=1
        )

        tqdm.pandas(desc="Dep Tree Clean Form    ")
        df["dependency_tree_clean_norm"] = df.progress_apply(
            lambda x: self.create_clean_tree_norm(x["dependency_tree_norm"]), axis=1
        )

        tqdm.pandas(desc="Root Index             ")
        df["root_norm"] = df.progress_apply(
            lambda x: self.get_root_norm(x["dependency_tree_norm"]), axis=1
        )

        tqdm.pandas(desc="Dependency Labels      ")
        df["dependency_norm"] = df.progress_apply(
            lambda x: [i for i, _, _ in x["dependency_tree_norm"][1:]], axis=1
        )

        tqdm.pandas(desc="Tokens                 ")
        df["tokens"] = df.sent.progress_apply(
            lambda x: [token.text for token in x.tokens]
        )

        tqdm.pandas(desc="Tokens (Normalized)    ")
        df["tokens_norm"] = df.progress_apply(
            lambda x: self.get_tokens_norm(x["sent"]), axis=1
        )

        df.drop(columns="sent", inplace=True)

        if save:
            with open(save_path, "wb") as f:
                pickle.dump(df, f)
        return df

    @staticmethod
    def extract_xpos(text):
        res = re.search(r".+?(?=\|)", text)
        if res is not None:
            return res.group()
        res = text
        return res

    @staticmethod
    def get_RGAT_aspects(row):
        idxs = row["aspect_idx"]
        if type(idxs) is int:
            start = idxs
            end = idxs + 1
        else:
            start = idxs[0]
            end = idxs[-1] + 1

        return {
            "term": row["tokens_norm"][start:end],
            "from": int(start),
            "to": int(end),
            "polarity": "negative" if row["sentiment"] == 0 else "positive",
        }

    def get_RGAT_format(self, df):
        if "sent" not in df.columns:
            tqdm.pandas(desc="Stanza Sentence Parsing")
            df["sent"] = df.text.progress_apply(lambda x: self.sent_parse(x))
        return [
            {
                "token": row["tokens_norm"],
                "pos": [self.extract_xpos(t["xpos"]) for t in row["sent"].to_dict()],
                "head": [str(t["head"]) for t in row["sent"].to_dict()],
                "deprel": [t["deprel"] for t in row["sent"].to_dict()],
                "aspects": [self.get_RGAT_aspects(row)],
            }
            for _, row in df.iterrows()
        ]
