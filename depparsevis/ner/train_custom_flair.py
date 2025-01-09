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
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

# 4. initialize embeddings
embeddings = TransformerWordEmbeddings(
    model="xlm-roberta-large",
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)


# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=label_dict,
    tag_type="ner",
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. run fine-tuning
trainer.fine_tune(
    "resources/taggers/sota-ner-flert",
    learning_rate=5.0e-6,
    mini_batch_size=4,
    mini_batch_chunk_size=1,
    max_epochs=10,
)

# # 7. start training
# trainer.train(
#     "../../models/ner-flair-basic",
#     train_with_dev=False,
#     mini_batch_size=16,
#     max_epochs=50,
# )
