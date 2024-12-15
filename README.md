# AutoLLM-CARD: Towards a Description and Landscape of Large Language Models

# Abstract 
With the rapid growth of the Natural Language Processing (NLP) field, a vast variety of Large Language Models (LLMs) continue to emerge for diverse NLP tasks. As more papers are published, researchers and developers face the challenge of information overload. Thus, developing a system that can automatically extract and organise key information about LLMs from academic papers is particularly important. The standard format for documenting information about LLMs is the LLM model card (\textbf{LLM-Card}). We propose a method for automatically generating LLM model cards from scientific publications. We use Named Entity Recognition (\textbf{NER}) and Relation Extraction (\textbf{RE}) methods that automatically extract key information about LLMs from the papers, helping researchers to access information about LLMs efficiently. These features include model \textit{licence}, model \textit{name}, and model \textit{application}. With these features, we can form a model card for each paper. We processed 106 academic papers by defining three dictionaries -- LLM's name, licence, and application. 11,051 sentences were extracted through dictionary lookup, and the dataset was constructed through manual review of the final selection of 129 sentences with a link between the name and the \textit{licence}, and 106 sentences with a link between the model name and the \textit{application}. The resulting resource is relevant for LLM card illustrations using relational knowledge graphs. Our code and findings can contribute to automatic LLM card generation.

# Github instructions
<table>
  <tr>
    <th>Files</th>
    <th>Content</th>
  </tr>
  <tr>
    <td>my_combined_graph</td>
    <td>the graphs on (model name, application, licence)</td>
  </tr>
  <tr>
    <td>llm_model_applications.csv</td>
    <td>data extracted on (models, applications)</td>
  </tr>
  <tr>
    <td>license_sentences.csv</td>
    <td>data extracted on (models, license)</td>
  </tr>
  <tr>
    <td>code.ipynb</td>
    <td>dependency parsing and pdf processing</td>
  </tr>
  <tr>
    <td>cleaned_extracted_license_relationships.csv</td>
    <td>triples extracted on (models, license)</td>
  </tr>
  <tr>
    <td>cleaned_extracted_llm_relationships.csv</td>
    <td>triples extracted on (models, application)</td>
  </tr>
</table>
my_combined_graph｜ the graphs on (model name, application, licence).

llm_model_applications.csv ｜ data extracted on (models, applications)

license_sentences.csv ｜	Data on (models, license)

code.ipynb	｜ Dependency parsing and pdf processing


# reference

@misc{tian2024autollmcarddescriptionlandscapelarge,
      title={AutoLLM-CARD: Towards a Description and Landscape of Large Language Models}, 
      author={Shengwei Tian and Lifeng Han and Goran Nenadic},
      year={2024},
      eprint={2409.17011},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.17011}, 
}
