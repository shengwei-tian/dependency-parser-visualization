import os
import random
from ast import literal_eval

import pandas as pd
import spacy
from tqdm.auto import tqdm

labels_to_tag = {
    "O": 0,
    "B-Model": 1,
    "I-Model": 2,
    "B-License": 3,
    "I-License": 4,
    "B-Application": 5,
    "I-Application": 6,
}

tags_to_label = {v: k for k, v in labels_to_tag.items()}


if __name__ == "__main__":
    input_file = "../../data/paper_analysis_20250103_113938.csv"
    output_folder = "../../data"

    # Load spacy model, used same model as in code.ipynb
    nlp = spacy.load("en_core_web_sm")

    # Load annotations
    df = pd.read_csv(input_file)

    lines = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        doc = nlp(row.sentence)
        tags = literal_eval(row.tags)

        line = ""
        for token, tag in zip(doc, tags):
            line = (
                line + token.text + "\t" + token.tag_ + "\t" + tags_to_label[tag] + "\n"
            )

        line = line + "\n"

        lines.append(line)

    # --- Randomly shuffle the data
    random.shuffle(lines)

    # --- Based on previous split sizes 7-2-1
    train_ratio = 0.7
    val_ratio = 0.2

    train_size = int(train_ratio * len(lines))
    val_size = int(val_ratio * len(lines))

    train_lines = lines[:train_size]
    val_lines = lines[train_size : train_size + val_size]
    test_lines = lines[train_size + val_size :]

    # --- Write them out to three separate files
    with open(os.path.join(output_folder, "train_column_dataset.txt"), "w") as fp:
        fp.writelines(train_lines)

    with open(os.path.join(output_folder, "val_column_dataset.txt"), "w") as fp:
        fp.writelines(val_lines)

    with open(os.path.join(output_folder, "test_column_dataset.txt"), "w") as fp:
        fp.writelines(test_lines)
