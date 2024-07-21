import os
import torch
import ankh

# Function to extract protein sequence from fasta file
def get_protein_sequence(protein_id, fasta_path):
    fasta_file = os.path.join(fasta_path, f"{protein_id}.fasta")
    
    # Assuming each fasta entry is a single line
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        # Concatenate all lines (excluding the header) to get the sequence
        sequence = ''.join(lines[1:]).replace('\n', '')
    
    return sequence

# Function to perform embedding and save to .pt file
def embed_and_save(protein_sequence, prot_save_path, pdb_id, motif_no_embed_file):
    model, tokenizer = ankh.load_base_model()
    model.eval()

    # Check the length of the protein sequence
    if len(protein_sequence) <= 10000:
        # Tokenize and obtain embedding for the entire sequence
        outputs = tokenizer.batch_encode_plus([list(protein_sequence)],
                                              add_special_tokens=True,
                                              padding=True,
                                              is_split_into_words=True,
                                              return_tensors="pt")
        
        with torch.no_grad():
            embedding = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])

        # Remove special token (last token from the embedding)
        clean_embedding = embedding['last_hidden_state'][:, :-1, :]

        # Save the final embedding to a .pt file
        torch.save(clean_embedding, prot_save_path)
        print(f"Embedding for {pdb_id} is done")
    else:
        # If protein sequence length is greater than 10000, write pdb_id to the no_embedded_proteins.txt file
        with open(motif_no_embed_file, 'a') as no_embed_file:
            no_embed_file.write(f"{pdb_id}\n")
        print(f"Embedding for {pdb_id} is skipped")

# Read the input file line by line
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"
fasta_path = "/home/auber/merabet/prot_sequences"
prot_output_path = "/home/auber/merabet/prot_sequence_embeddings"
motif_output_path = "/home/auber/merabet/PU_nPU_embeddings"
motif_no_embed_file_path = "/home/auber/merabet/no_embedded_motifs.txt"
no_embed_proteins_file_path = "/home/auber/merabet/no_embedded_proteins.txt"

# Dictionary to store unique protein sequences and their embeddings
unique_protein_sequences = {}

# Embed protein sequences
with open(input_file_path, 'r') as input_file:
    for line in input_file:
        # Split the tab-delimited line
        columns = line.strip().split('\t')

        # Extract relevant information from the columns
        motif_id = columns[0]
        Uniprot_id = columns[2].split('_')[0]
        PDB_id = columns[0].split('_')[0]
        start_position = int(columns[2].split('_')[1])
        end_position = int(columns[2].split('_')[2])

        # Get the protein sequence from the fasta file
        protein_sequence = get_protein_sequence(Uniprot_id, fasta_path)

        # Check if the protein sequence has been embedded before
        if PDB_id not in unique_protein_sequences:
            # Generate a unique filename based on the protein ID
            prot_output_filename = f"{PDB_id}.pt"
            prot_output_file_path = os.path.join(prot_output_path, prot_output_filename)

            # Embed the sequence and save to .pt file, or write PDB_id to no_embedded_proteins.txt
            embed_and_save(protein_sequence, prot_output_file_path, PDB_id, no_embed_proteins_file_path)

            # Store the protein sequence and its embedding in the dictionary
            unique_protein_sequences[PDB_id] = {
                'sequence': protein_sequence,
                'embedding_path': prot_output_file_path
            }

# Embed motifs
with open(input_file_path, 'r') as input_file:
    for line in input_file:
        # Split the tab-delimited line
        columns = line.strip().split('\t')

        # Extract relevant information from the columns
        motif_id = columns[0]
        PDB_id = columns[0].split('_')[0]
        start_position = int(columns[2].split('_')[1])
        end_position = int(columns[2].split('_')[2])

        # Search for the protein embedding file in the specified folder
        prot_embedding_file_path = os.path.join(prot_output_path, f"{PDB_id}.pt")

        # Check if the protein embedding file exists
        if os.path.exists(prot_embedding_file_path):
            # Load the protein embedding
            prot_embedding = torch.load(prot_embedding_file_path)

            # Extract the corresponding part of the embedding
            motif_embedding = prot_embedding[:, start_position - 1:end_position, :]

            # Generate a unique filename based on the motif ID
            motif_output_filename = f"{motif_id}.pt"
            motif_output_file_path = os.path.join(motif_output_path, motif_output_filename)

            # Save the motif embedding to a .pt file
            torch.save(motif_embedding, motif_output_file_path)
            print(f"Embedding for {motif_id} is done")
        else:
            # If protein embedding is not available, write motif_id to the no_embedded_motifs.txt file
            with open(motif_no_embed_file_path, 'a') as no_embed_file:
                no_embed_file.write(f"{motif_id}\n")
            print(f"Embedding for {motif_id} is skipped")

print("end")