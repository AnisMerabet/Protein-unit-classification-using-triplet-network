import os

def process_motifs(file_path, fasta_directory):
    with open(file_path, 'r') as file:
        motifs = [line.strip().split()[0] for line in file]

    for motif in motifs:
        # Extract the first term
        motif_name = motif.split('_')[0]

        # Search for .fasta file
        fasta_path = os.path.join(
            "/dsimb/auber/gelly/PROJECTS/PROJECT_PROTEIN_UNITS_ghouzam/wagram_peeling/data/PDBs_Clean",
            motif_name,
            "Peeling",
            f"{motif}.fasta"
        )

        # Save .fasta file into the new directory
        destination_path = os.path.join(fasta_directory, f"{motif}.fasta")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        os.system(f"cp {fasta_path} {destination_path}")

    return motifs, fasta_directory

# Call the function
motifs, fasta_directory = process_motifs('/home/auber/merabet/PU_nPU_list.txt', '/home/auber/merabet/PU_nPU_fasta_files')
