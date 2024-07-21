# Going further in Protein Units Recognition and prediction of the classification using embedding and triplet networks

Protein Units (PUs) serve as an intermediate level between secondary structures and protein domains. They are generated using Peeling algorithm and classified using Markov Clustering Algorithm (MCL) [1]. The Classified PUs are named Core Protein Units (CPUs) and the non-classified PUs are named Non-Core Protein Units (NCPUs). The CPUs are classified according to five different hierarchical levels: Class, Architecture, Topology and NanoFold and Family.
This project aims to use a protein language model (Ankh In this case) to generate embeddings for PUs then combine oversampling and few shot learning strategies to build a model able to distinguish between CPUs and NCPUs. The latter model successively fine-tuned to be able to predict the categories of CPUs according to Class, Architecture, Topology and NanoFold levels. The performance of the 5 model are compared to models based on one-hot encoding of the protein sequences.

> [1](https://academic.oup.com/bioinformatics/article/22/2/129/424567?login=false):
J.-C. Gelly, A. G. de Brevern, and S. Hazout, “‘Protein Peeling’: an approach for splitting a 3D protein structure into compact fragments,” Bioinforma. Oxf. Engl., vol. 22, no. 2, pp. 129–133, Jan. 2006, doi: 10.1093/bioinformatics/bti773.

Here we provide the general workflow of the project along with a general explanation for each part of the code.

## Data filtration and generation of the data list

The code `make_data_list.py` allow to to create lists for NCPUs `non_PU_list.txt`, and CPUs `PU_list.txt` and combine them into one list `PU_nPU_list.txt`. It converts PDB IDs into Uniprot IDs using a modified version of pdb2uniprot package `pdb2uniprot_modified.py`. The source code of the package was modified to correct some issues regarding duplicates and the order of the output.

> [pdb2uniprot](https://github.com/johnnytam100/pdb2uniprot.git)

It then downloads fasta file of complete protein sequence for each Uniprot ID and perform global local search to map each PU to its corresponding protein sequence and get nucleotide coordinates of the mapping. The used function is `glsearch36` from `fasta36` package.

> [fasta36](https://github.com/wrpearson/fasta36)

It handles overlapping classifications at the NanoFold level and filter the data, notably by removing duplicates. The last part of the code filters the data to remove PUs for which the embedding was not possible. Thus, this part should be run after running the code `generate_embedding_ankh.py`.
The file `PU_nPU_list.txt` is ultimately organized into 9 tab-delimited columns:
[PDB ID & coordinates] [PDB ID] [Uniprot ID & coordinates] [Uniprot ID] [Type] [Class] [Architecture] [Topology] [NanoFold]

## Generate embeddings and prepare fasta files to facilitate one-hot encoding

The code `generate_embedding_ankh.py` is used to generate the embedding of the complete protein sequences and extract the embedding for the PUs. It used the base model from the package Ankh [2].

> [2](https://arxiv.org/abs/2301.06568):
A. Elnaggar et al., “Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling.” arXiv, Jan. 16, 2023. doi: 10.48550/arXiv.2301.06568.

> [Ankh](https://github.com/agemagician/Ankh)

The code `extract_fasta_files.py` is used to gather all fasta files of PUs in one folder to facilitate the one-hot encoding afterwards.

## Data exploration

The code `data_exploration.py` performs diverse exploration strategies on the dataset by plotting some figures and documenting some outputs in .txt files.

## Dataset preparation

The code `generate_datasets.py` allow to divide the dataset into train, validation and test datasets having 60%, 15% and 15% of the data respectively. Then following an oversampling strategy, it generates triplets (consisting of anchor, positive example and negative example) for the models predicting Type, Class, Architecture, Topology and NanoFold. It generates 15 .txt files containing the triplets. The files also mention the category of the anchor PU and the weights (usage of weights was deprecated and replaced with the oversampling strategy).

## Model construction and evaluation

The codes `triplet_network_type.py`, `triplet_network_class.py`, `triplet_network_architecture.py`, `triplet_network_topology.py`, `triplet_network_nanofold.py` allow to construct 5 models consisting of triplet networks with two layers (GRU layer followed by a fully connected layer) predicting categories of PUs according to Type (CPUs and NCPUs), Class, Architecture, Topology and NanoFold respectively. All these models are based on Ankh-generated embeddings. The fine-tuning was successively conducted through the hierarchical levels.
The codes `triplet_network_one_hot_type.py`, `triplet_network_one_hot_class.py`, `triplet_network_one_hot_architecture.py`, `triplet_network_one_hot_topology.py`, `triplet_network_one_hot_nanofold.py` construct models similar to the previous models, but allow to perform one-hot encoding for the protein sequences of PUs instead of using Ankh-generated embeddings.
Each of the codes contain a model evaluation part allowing the calculation of many metrics, plot different figures and summarize results in .txt files.