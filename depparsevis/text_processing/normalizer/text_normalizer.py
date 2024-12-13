import html
import os
import re
from string import punctuation

import emoji
import hunspell
import nltk
import spacy
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm


class Normalizer:
    def __init__(self):

        nl_dic_path = os.path.join(
            os.path.dirname(__file__), "opentaal_hunspell/nl.dic"
        )
        nl_aff_path = os.path.join(
            os.path.dirname(__file__), "opentaal_hunspell/nl.aff"
        )
        en_dic_path = os.path.join(
            os.path.dirname(__file__), "opentaal_hunspell/en.dic"
        )
        en_aff_path = os.path.join(
            os.path.dirname(__file__), "opentaal_hunspell/en.aff"
        )

        if (
            (not os.path.isfile(nl_dic_path))
            or (not os.path.isfile(nl_aff_path))
            or (not os.path.isfile(en_dic_path))
            or (not os.path.isfile(en_aff_path))
        ):
            print("Make sure file paths are correct!")
            raise FileNotFoundError

        self.hobj_nl = hunspell.HunSpell(nl_dic_path, nl_aff_path)
        self.hobj_en = hunspell.HunSpell(en_dic_path, en_aff_path)
        self.nlp = spacy.load("nl_core_news_lg")

    def hunspell_check(self, tokens, lang="nl"):
        """Apply spell checking and stemmize if stem is True

        Args:
            tokens (list([str])): Tokens
            stem (bool, optional): Defaults to False.

        Returns:
            list([str]): Processed Tokens
        """
        # Process a spell check and stemmize if stem=True
        hobj = self.hobj_en if lang == "en" else self.hobj_nl

        processed_tokens = []
        for token in tokens:
            if not hobj.spell(token):
                suggestions = hobj.suggest(token)
                if len(suggestions) >= 1:
                    token = suggestions[0]
                # else:
                #    continue # If the word is not in word list then remove skip it
            processed_tokens.append(token)
        return processed_tokens

    @staticmethod
    def clean_lines(text):
        """Removes \n, \r\n, \r\r\n from the text and replace them with a space

        Args:
            text (str): String text multiple lines

        Returns:
            str: Newlines replaced text
        """
        lines = list(filter(lambda l: l != "", text.splitlines()))
        return " ".join(lines)

    @staticmethod
    def html_like_tag_replace(text, replace=""):
        """Removes html-like tags from text.

        Args:
            text (str): text to be processed
            replace (str, optional): Placeholder for html-like tags. Defaults to "".

        Returns:
            str: Html-like tags removed text
        """
        # Remove html tag-like part from previously seen two record.
        return re.sub(r"<.+?>", replace, text, flags=re.IGNORECASE)

    @staticmethod
    def url_replace(text, replace="<URL>"):
        """Replaces URL's with given string

        Args:
            text (str): text to be processed
            replace (str, optional): Placeholder for URL's. Defaults to "<URL>".

        Returns:
            str: URL's removed text
        """
        return re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", replace, text)

    @staticmethod
    def un_html(text):
        """Fixes parsing errors such as:
            - I&#039;m : I'm
            - I&#039;ve : I'have

        Args:
            text (str): Text to be processed

        Returns:
            str: Processed text
        """
        return html.unescape(text)

    @staticmethod
    def expand_abbreviations(text):
        """Expands abbreviations in the text.
        Reference: https://www.expatica.com/nl/living/integration/dutch-abbreviations-100677/

        Args:
            text (str): Text includes abbreviations

        Returns:
            str: Text with expanded abbreviations
        """
        afkortings_dict = {
            "a.s": "aanstaande",
            "aso": "asociaal",
            "begr": "begraven",
            "beh": "behalve",
            "btw": "Belasting Toegevoegde Waarde",
            "ca": "circa",
            "dd": "de dato",
            "dir": "directeur",
            "dwz": "dat wil zeggen",
            "d.w.z": "dat wil zeggen",
            "dwz.": "dat wil zeggen",
            "geb": "geboren",
            "gesch": "gescheiden",
            "igvn": "in geval van nood",
            "i.g.v.n": "in geval van nood",
            "iha": "in het algemeen",
            "i.h.a": "in het algemeen",
            "iig": "in ieder geval",
            "i.i.g": "in ieder geval",
            "itt": "in tegenstelling tot",
            "i.t.t": "in tegenstelling tot",
            "miv": "met ingang van",
            "m.i.v": "met ingang van",
            "mv": "meervoud",
            "muv": "met uitzondering van",
            "m.u.v": "met uitzondering van",
            "m.vr.gr": "met vriendelijke groeten",
            "nav": "naar aanleiding van",
            "n.a.v": "naar aanleiding van",
            "notk": "nader overeen te komen",
            "n.o.t.k": "nader overeen te komen",
            "o.a": "onder andere",
            "o.l.v": "onder leiding van",
            "o.m": "onder meer",
            "ong": "ongeveer",
            "oorspr": "oorspronkelijk",
            "overl": "overleden",
            "tav": "ter attentie van",
            "t.a.v": "ter attentie van",
            "teab": "tegen elk aannemelijk bod",
            "t.e.a.b": "tegen elk aannemelijk bod",
            "tnv": "ten name van",
            "t.n.v": "ten name van",
            "twv": "ter waarde van",
            "t.w.v": "ter waarde van",
            "v.a": "vanaf",
            "vnl": "voornamelijk",
            "zgn": "zogenaamd",
            "zoz": "zie ommezijde",
            "z.o.z": "zie ommezijde",
            "aub": "alstublieft",
            "a.u.b": "alstublieft",
            "enz": "enzovoorts",
            "prof": "professor",
            "dmv": "door middel van",
            "d.m.v": "door middel van",
            "dhr": "de heer",
            "o.k": "oké",
            "ok": "operatie kamer",
            "nvt": "niet van toepassing",
            "n.vt": "niet van toepassing",
            "n.v.t": "niet van toepassing",
            "v/d": "van de",
            "v.d": "van de",
            "vd": "van de",
            "c.q": "respectievelijk",
            "bv": "bijvoorbeeld",
            "bijv": "bijvoorbeeld",
            "vb": "voorbeeld",
            "plm": "plusminus",
            "i.v.m": "in verband met",
            "ivm": "in verband met",
            "tbv": "ten behoeve van",
            "t.b.v": "ten behoeve van",
            "seh": "spoedeisende hulp",
            "m.b.t": "met betrekking tot",
            "mbt": "met betrekking tot",
            "tel": "telefoon",
            "mevr": "mevrouw",
            "mw": "mevrouw",
            "bsn": "burgerservicenummer",
            "incl": "inclusief",
            "exc": "exclusief",
            "excl": "exclusief",
            "m/v": "man/vrouw",
            "t/m": "tot en met",
            "tgv": "ten gevolge van",
            "t.g.v": "ten gevolge van",
            "info": "informatie",
            "i.p.v": "in plaats van",
            "ipv": "in plaats van",
            "afd": "afdeling",
            "etc": "et cetera",
            "i.c": "intensieve zorg",
            "ic": "intensieve zorg",
            "nicu": "neonatale intensive care unit",
            "p.s": "postscriptum",
            "ps": "postscriptum",
            "epd": "elektronisch patiëntendossier",
            "lumc": "Leids Universitair Medisch Centrum",
            "t.o.v": "ten opzichte van",
            "tov": "ten opzichte van",
            "nl": "namelijk",
            "ehbo": "eerste hulp bij ongelukken",
            "ct": "computertomografie",
            "ct-scan": "computertomografie",
            "x-ray": "röntgenfoto",
            "i.a": "in afwezigheid",
            "ia": "in afwezigheid",
            "kno": "keel-, neus- en oorheelkunde",
            "pacu": "post-abortus zorgeenheid",
            "eea": "een en ander",
            "e.e.a": "een en ander",
            "ivf": "in-vitrofertilisatie",
            "i\.v\.f": "in-vitrofertilisatie",
            "d.r": "doctor",
            "dr": "doctor",
            "evt": "eventueel",
            "alg": "algemeen",
            "v.h": "van het",
            "vh": "van het",
            "j.l": "jongstleden",
            "jl": "jongstleden",
            "aoa": "acute opname afdeling",
            "a.o.a": "acute opname afdeling",
            "mdl": "maag-, darm- en leverziekten",
            "m.d.l": "maag-, darm- en leverziekten",
            "voh": "verpleegafdeling oncologie en hematologie",
            "v.o.h": "verpleegafdeling oncologie en hematologie",
            "h a m": "hygiëne, arbeid en millieu",
            "h.a.m": "hygiëne, arbeid en millieu",
            "ham": "hygiëne, arbeid en millieu",
            "li": "links",
            "pa": "psychiatrische afdeling",
            "p.a": "psychiatrische afdeling",
            "1e": "eerste",
            "2e": "tweede",
            "3e": "derde",
            "4e": "vierde",
            "5e": "vijfde",
            "6e": "zesde",
            "7e": "zevende",
            "8e": "achtste",
            "9e": "negende",
            "10e": "tiende",
            "o.i.d": "of iets dergelijks",
            "oid": "of iets dergelijks",
        }

        mod = list(map(lambda x: re.sub(r"\.", r"\\.", x), afkortings_dict.keys()))
        afkortings_re = re.compile(r"\b(%s)\b" % "|".join(mod), flags=re.I)

        def replace(match):
            return afkortings_dict[match.group(0).lower()]

        return afkortings_re.sub(replace, text)

    @staticmethod
    def correct_apos(text):
        text = re.sub(r"\’", "'", text)
        text = re.sub(r"\‘", "'", text)
        text = re.sub(r"\`", "'", text)
        return re.sub(r"\´", "'", text)

    @staticmethod
    def expand_contractions(text):
        """Expands contraction in the text

        Args:
            text (str): Text includes contractions

        Returns:
            str: Text with expanded contractions
        """
        contractions_dict = {
            "t'is": "het is",
            "da's": "dat is",
            "di's": "dit is",
            "d'r": "de haare",
            "h'r": "haar",
            "z'n": "zijn",
            "m'n": "mijn",
            "zo'n": "zo een",
        }
        contractions_re = re.compile("(%s)" % "|".join(contractions_dict.keys()))

        def replace(match):
            return contractions_dict[match.group(0)]

        return contractions_re.sub(replace, text)

    @staticmethod
    def split_slashes(text):
        """Text includes slashes ('/'). It expands slashes with spaces to correctly tokenize later.

        Args:
            text (str): Text to be processed

        Returns:
            str: Processed text
        """
        return re.sub(r"/", " / ", text)

    @staticmethod
    def number_replace(text, replace="<NUMBER>"):
        """Replaces numbers with specified placeholder

        Args:
            text (str): Text to be processed
            replace (str, optional): Placeholder. Defaults to "<NUMBER>".

        Returns:
            str: Processed text
        """
        return re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", replace, text)

    @staticmethod
    def pucnt_repeat_replace(text, replace=" <REPEAT>"):
        """Reduces the repeated punctuations to 1 and adds a placeholder next to it

        Args:
            text (str): Text to be processed
            replace (str, optional): Placeholder. Defaults to ' <REPEAT>'.

        Returns:
            str: Processed text
        """
        # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
        return re.sub("([!?.]){2,}", f"\g<1>{replace}", text)

    @staticmethod
    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        return tag_dict.get(tag, wordnet.NOUN)

    def preprocessor(
        self,
        data,
        text_col="text",
        lang_col="lang",
        removePuntc=True,
        spellcheck=True,
        lemmatize=True,
        filter_length=True,
        lower=False,
    ):

        sentence = data[text_col]

        # 1. Remove spaces from beginning and the end of sentence
        sentence = sentence.strip()

        ## Regex part before tokenization
        # 2. Fix parsing errors
        sentence = self.un_html(sentence)

        # 3. Correct apostrophes
        sentence = self.correct_apos(sentence)

        # 4. Clean newlines
        sentence = self.clean_lines(sentence)

        # 5. Remove URLs
        sentence = self.url_replace(sentence, replace="[url]")

        # 6. Remove <...>
        # sentence = self.html_like_tag_replace(sentence, replace="")

        # 7. Before removing punctuations change possible signs into words
        sentence = re.sub("\$", " dollar ", sentence)
        sentence = re.sub("£", "pond sterling", sentence)
        sentence = re.sub("\%", " percent ", sentence)
        sentence = re.sub("\&", " en ", sentence)
        sentence = re.sub("\+_", " plusminus ", sentence)
        # sentence = re.sub("\[naam\]", "<NAAM>", sentence)
        # sentence = re.sub("\[datum\]", "<DATUM>", sentence)
        # sentence = re.sub("\[email\]", "<EMAIL>", sentence)
        # sentence = re.sub("\[tel\]", "<TEL>", sentence)
        sentence = re.sub("C.{1,5}Q", "[locatie]", sentence)

        # 8. Expand contractions
        sentence = self.expand_contractions(sentence)

        # 9. Expand abbreviations
        sentence = self.expand_abbreviations(sentence)

        # 10. Split slashes
        sentence = self.split_slashes(sentence)

        # 11. Remove numbers
        # sentence = self.number_replace(sentence, replace="<NUMMER>")

        # 12. Demojize sentence
        sentence = emoji.demojize(sentence)

        # 13. Remove punctuations
        if removePuntc:
            sentence = sentence.translate(
                sentence.maketrans(punctuation, " " * len(punctuation))
            )

        # 12. Tokenize
        tokens = nltk.word_tokenize(text=sentence, language="dutch")

        try:
            lang = data[lang_col]
        except KeyError:
            lang = "nl"

        if lang == "en":
            # 13. Spell Check (EN)
            if spellcheck:
                tokens = self.hunspell_check(tokens, lang="en")

            # 14. Lemmatize (EN)
            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                tokens = [
                    lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in tokens
                ]
        else:
            # 13. Spell Check (NL)
            if spellcheck:
                tokens = self.hunspell_check(tokens, lang="nl")
                sentence = " ".join(tokens)

            # 14. Lemmatize (NL)
            if lemmatize:
                for doc in self.nlp.pipe([sentence]):
                    tokens = [t.lemma_ for t in doc]

        # 16. Filter tokens by length
        if filter_length:
            sentence = " ".join(
                [w.lower() if lower else w for w in tokens if (len(w) > 1)]
            )
        else:
            sentence = " ".join([w.lower() if lower else w for w in tokens])

        return sentence

    def normalize(self, df, text_col="text", lang_col="lang"):
        """[summary]

        Args:
            df (pandas.DataFrame): DataFrame with the columns (text and lang) to be normalized
            text_col (str, optional): Column name to be normalized. Defaults to 'text'.
            lang_col (str, optional): Column name shows which languege is the text column. Defaults to 'lang'.

        Returns:
            pandas.Series: Normalized text as pandas.Series
        """
        tqdm.pandas()
        return df.progress_apply(
            lambda x: self.preprocessor(x, text_col, lang_col), axis=1
        )

    def parametrized_normalization(
        self,
        df,
        text_col="text",
        lang_col="lang",
        removePuntc=False,
        spellcheck=False,
        lemmatize=False,
        filter_length=False,
        lower=False,
    ):
        """[summary]

        Args:
            df (pandas.DataFrame): DataFrame with the columns (text and lang) to be normalized
            text_col (str, optional): Column name to be normalized. Defaults to 'text'.
            lang_col (str, optional): Column name shows which languege is the text column. Defaults to 'lang'.

        Returns:
            pandas.Series: Normalized text as pandas.Series
        """
        tqdm.pandas()
        return df.progress_apply(
            lambda x: self.preprocessor(
                x,
                text_col,
                lang_col,
                removePuntc=removePuntc,
                spellcheck=spellcheck,
                lemmatize=lemmatize,
                filter_length=filter_length,
                lower=lower,
            ),
            axis=1,
        )
