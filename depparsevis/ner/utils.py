import numpy as np
import spacy
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import sent_tokenize
from numpy.random import randint
from spacy import displacy


class ModelCardTagger:
    def __init__(self, model_path="../../models/ner-flair-basic/final-model.pt"):
        self.model = SequenceTagger.load(model_path)

    def annotate_entities(self, sent):
        sents = sent_tokenize(sent)

        aspects = []
        for s in sents:
            sentence = Sentence(s)
            self.model.predict(sentence)
            aspects.append(
                [
                    (l.data_point.start_position, l.data_point.end_position, l.value)
                    for l in sentence.labels
                ]
            )

        lengths = [len(s) for s in sents]
        aspects2 = []
        ll = 0
        for l, asps in zip(lengths, aspects):
            aspects2.append([(s + ll, e + ll, a) for (s, e, a) in asps])
            ll = ll + l + 1

        assert sent == " ".join(sents)

        return [pair for ss in aspects2 for pair in ss]

    def annotate_entities_one_sentence(self, sent):
        sentence = Sentence(sent)
        self.model.predict(sentence)
        return [
            (l.data_point.start_position, l.data_point.end_position, l.value)
            for l in sentence.labels
        ]

    def predict_and_display(self, sent):
        sentence = Sentence(sent)
        self.model.predict(sentence)
        return display_ner_flair(sentence)


def show(df, model, keer=5):
    for _ in range(keer):
        ix = randint(len(df))
        print(ix)
        sent = df.loc[ix, "text"]

        sentence = Sentence(sent)

        # predict NER tags
        model.predict(sentence)

        # print sentence with predicted tags
        display_ner_flair(sentence)
        display_ner_true(df, ix)
        print("\n\n")


def show_these(df, model, ixs):
    for ix in ixs:
        sent = df.loc[ix, "text"]

        sentence = Sentence(sent)

        # predict NER tags
        model.predict(sentence)

        # print sentence with predicted tags
        display_ner_flair(sentence)
        display_ner_true(df, ix)
        print("\n\n")


def display_ner_flair(sentence):
    colors = {
        "Model": "#7aecec",
        "Application": "#ff9561",
        "Licence": "#c887fb",
    }

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence.text)

    try:
        ents = []
        for label in sentence.labels:
            start = label.data_point.start_position
            end = label.data_point.end_position
            ents.append(doc.char_span(start, end, label.value))

        doc.ents = ents
    except:  # noqa
        return print(doc.text)
    return displacy.render(doc, style="ent", jupyter=True, options={"colors": colors})


def display_ner_true(df, ix):
    colors = {
        "Model": "#7aecec",
        "Application": "#ff9561",
        "Licence": "#c887fb",
    }

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(df.loc[ix, "text"])

    asp = df.loc[ix, "aspect"][0]

    start = asp[0]
    end = asp[1]
    value = asp[2]

    doc.ents = [doc.char_span(start, end, value)]

    return displacy.render(doc, style="ent", jupyter=True, options={"colors": colors})


def prediction_accuracy(df, model, check_overlap=True):

    def is_overlapping(a, b, c, d):
        return (c - b) < 0 and (a - d) < 0

    preds = []
    for ix in range(len(df)):

        sent = df.loc[ix, "text"]

        sentence = Sentence(sent)

        # predict NER tags
        model.predict(sentence)

        asp = df.loc[ix, "aspect"][0]
        a = asp[0]
        b = asp[1]
        aspect = asp[2]

        ll = [l.value for l in sentence.labels]  # noqa

        check1 = aspect.upper() in ll
        if check1:
            if check_overlap:
                ind = ll.index(aspect.upper())
                label = sentence.labels[ind]
                c = label.data_point.start_position
                d = label.data_point.end_position
                check2 = is_overlapping(a, b, c, d)
                if check2:
                    preds.append(1)
                else:
                    preds.append(0)
            else:
                preds.append(1)
        else:
            preds.append(0)

    return np.array(preds).mean()
