from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# define columns
columns = {0: "text", 1: "pos", 2: "ner"}

# this is the folder in which train, test and dev files reside
data_folder = "../../data"

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(
    data_folder,
    columns,
    train_file="train_column_dataset.txt",
    test_file="train_column_dataset.txt",
    dev_file="train_column_dataset.txt",
)

# 2. what label do we want to predict?
label_type = "ner"

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embeddings
embeddings = TransformerWordEmbeddings(
    "wietsedv/bert-base-dutch-cased", allow_long_sentences=True
)


# 5. initialize sequence tagger
tagger: SequenceTagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type=label_type,
)

# 6. initialize trainer
trainer: ModelTrainer = ModelTrainer(tagger, corpus)


# 7. start training
trainer.train(
    "../../models/ner-flair-basic",
    train_with_dev=True,
    mini_batch_size=16,
    max_epochs=150,
)
