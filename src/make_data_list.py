# Make sure you do "module load fasta" in your terminal prior to running this code
## Create non-PU list -------------------------------------------------------------------------------------------------------------------------
with open('/dsimb/auber/gelly/PROJECTS/PU_CLASSIFICATIONS/LIST_OF_PUs/SET_PUs_NON_CORES.txt', 'r') as input_file, open('/home/auber/merabet/non_PU_list.txt', 'w') as output_file:
    for line in input_file:
        line = line.strip()
        components = line.split('_')
        first_component = components[0]
        output_file.write(f"{line}\t{first_component}\tnon_PU\n")


## Create PU list -----------------------------------------------------------------------------------------------------------------------------
with open('/dsimb/auber/gelly/PROJECTS/PU_CLASSIFICATIONS/LISTS_CATFS_PU_PROTOTYPE_with_fused_famillies', 'r') as input_file, open('/home/auber/merabet/PU_list.txt', 'w') as output_file:
    for line in input_file:
        line = line.strip().replace(':', '')  # Remove ":" from the line
        fields = line.split()
        
        # Extract the names of the PUs
        names_to_duplicate = [field for field in fields if '_' in field and ':' not in field]
        
        # Check if the third column has only one letter
        if len(fields) > 2 and len(fields[2]) == 1:
            continue  # Skip the line if the third column has only one letter
        
        # Duplicate the line for each name and write to the output file
        for name in names_to_duplicate:
            # Extract the first component of the names of the PUs
            first_component = name.split('_')[0]
            
            # Add the two new columns and shift the existing columns
            new_line = "{}\t{}\tPU\t{}".format(name, first_component, '\t'.join(fields[0:-len(names_to_duplicate)-1]))
            output_file.write("{}\n".format(new_line))

# Combine PU and non-PU lists into one file
with open('/home/auber/merabet/non_PU_list.txt', 'r') as non_pu_file, open('/home/auber/merabet/PU_list.txt', 'r') as pu_file, open('/home/auber/merabet/PU_nPU_list.txt', 'w') as output_file:
    # Write the lines from non_PU_list.txt
    for line in non_pu_file:
        output_file.write(line)

    # Write the lines from PU_list.txt
    for line in pu_file:
        output_file.write(line)


## Convert PDB IDs into Uniprot IDs -----------------------------------------------------------------------------------------------------------
# To know about the method: https://nbviewer.org/github/fomightez/PDBrenum/blob/master/chainID_mapping_to_UniProt_id_demo.ipynb
# git clone https://github.com/johnnytam100/pdb2uniprot.git
# I modified the original code to fix some issues: avoid duplicates and keep the same order as in the input file.
# Create CSV containing PDB IDs in the right format
input_file_path = '/home/auber/merabet/PU_nPU_list.txt'
output_csv_path = '/home/auber/merabet/PDB_IDs.csv'

with open(input_file_path, 'r') as input_file, open(output_csv_path, 'w') as output_csv:
    # Write the CSV header
    output_csv.write("PDB_ID,CHAIN_ID\n")

    # Process each line of the input file
    for line in input_file:
        # Extract the PDB IDs
        components = line.split('\t')[1].strip()

        # Modify the component format
        pdb_id = components[:-1] + ',' + components[-1]

        # Write to the CSV file
        output_csv.write(f"{pdb_id}\n")

# Run pdb2uniprot_modified.py (my modified code) file like a commande line
import subprocess
# Define the command as a list of strings
command = [
    'python',
    'pdb2uniprot/pdb2uniprot_modified.py',
    '--input',
    'PDB_IDs.csv',
    '--pdb_col',
    'PDB_ID',
    '--chain_col',
    'CHAIN_ID'
]
# Use subprocess to convert PDB IDs into Uniprot IDs
subprocess.run(command)


## Some PDB IDs have many Uniprot IDs. The following code keeps only the first Uniprot ID for each PDB ID -------------------------------------
def filter_lines(input_file, output_file):
    with open(input_file, 'r') as input_file:
        lines = input_file.readlines()

    filtered_lines = []

    i = 0
    while i < len(lines):
        current_line = lines[i].strip().split(',')
        filtered_lines.append(lines[i])

        j = i + 1
        while j < len(lines):
            next_line = lines[j].strip().split(',')

            if current_line[:2] == next_line[:2] and current_line[2] != next_line[2]:
                j += 1
            else:
                break

        i = j

    with open(output_file, 'w') as output_file:
        output_file.writelines(filtered_lines)

# Filter to keep only the first Uniprot ID for each PDB ID
filter_lines('/home/auber/merabet/PDB_IDs_uniprot.csv', '/home/auber/merabet/filtered_PDB_IDs_uniprot.csv')


## Add Uniprot IDs into PU_nPU_list.txt as a new column ---------------------------------------------------------------------------------------
# Read the content of filtered_PDB_IDs_uniprot.csv and store the third column in a dictionary
import os
uniprot_dict = {}
with open("/home/auber/merabet/filtered_PDB_IDs_uniprot.csv", "r") as filtered_file:
    next(filtered_file)  # skip the header line
    for line in filtered_file:
        parts = line.strip().split(',')
        pdb_id_chain = f"{parts[0]}{parts[1]}"
        uniprot_dict[pdb_id_chain] = parts[2]

# Update PU_nPU_list.txt with the new column
input_filename = "/home/auber/merabet/PU_nPU_list.txt"
temp_filename = "/home/auber/merabet/temp_PU_nPU_list.txt"
# Update PU_nPU_list.txt with the new column
with open("/home/auber/merabet/PU_nPU_list.txt", "r") as input_file, open("/home/auber/merabet/temp_PU_nPU_list.txt", "w") as output_file:
    for line in input_file:
        # Split the existing columns
        columns = line.strip().split('\t')
        
        # Extract the second column (pdb_id_chain)
        pdb_id_chain = columns[1]
        
        # Get the corresponding uniprot value from the dictionary
        uniprot = uniprot_dict.get(pdb_id_chain, "NA")
        
        # Insert the new uniprot column as the third column
        new_line = '\t'.join([columns[0], columns[1], uniprot] + columns[2:])
        
        # Write the modified line to the output file
        output_file.write(new_line + '\n')

# Replace the original file with the modified file
os.replace("/home/auber/merabet/temp_PU_nPU_list.txt", "/home/auber/merabet/PU_nPU_list.txt")


## Remove all the lines where the Uniprot ID (3rd column) in NaN ------------------------------------------------------------------------------
import os
input_filename = "/home/auber/merabet/PU_nPU_list.txt"
temp_filename = "/home/auber/merabet/temp_PU_nPU_list.txt"
# Read PU_nPU_list.txt, filter out lines with "NaN" in the third column, and write to a temporary file
with open(input_filename, "r") as input_file, open(temp_filename, "w") as temp_file:
    for line in input_file:
        # Split the line into columns
        columns = line.strip().split('\t')
        
        # Check if the third column (uniprot) is not "NA"
        if columns[2] != "NaN":
            # Write the line to the temporary file
            temp_file.write(line)

# Replace the original file with the temporary file
os.replace(temp_filename, input_filename)


## Download the fasta file of complete protein sequence for each Uniprot ID -------------------------------------------------------------------
import os
import requests
from Bio import SeqIO

def download_fasta_from_file(file_path, output_folder="prot_sequences"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read UniProt IDs from the third column of the input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    uniprot_ids = [line.split()[2] for line in lines]

    # Remove duplicate UniProt IDs
    unique_uniprot_ids = list(set(uniprot_ids))

    base_url = "https://www.uniprot.org/uniprot/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

    error_count = 0

    for uniprot_id in unique_uniprot_ids:
        url = f"{base_url}{uniprot_id}.fasta"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            # Save the FASTA sequence to a file in the output folder
            output_path = os.path.join(output_folder, f"{uniprot_id}.fasta")
            with open(output_path, "w") as fasta_file:
                fasta_file.write(response.text)
            print(f"FASTA sequence for {uniprot_id} downloaded successfully to {output_path}.")
        else:
            print(f"Error downloading FASTA sequence for {uniprot_id}. Status code: {response.status_code}")
            error_count += 1

    print(f"\nTotal number of errors: {error_count}")

input_file_path = "/home/auber/merabet/PU_nPU_list.txt"
download_fasta_from_file(input_file_path)


## Use global local search to map each UP/non-UP to protein sequence and get nucleotide coordinates of the mapping ------------------------------------------------
import os
import shutil
import subprocess
import re

def process_glsearch_output(input_file_name, output_file_name, nan_counter):
    # Read the input file
    with open(input_file_name, 'r') as input_file:
        lines = input_file.readlines()

    # Extract library name and identifier from the first line
    first_line = lines[0]
    library_match = re.search(r'/([A-Za-z0-9]+)\.fasta', first_line)
    identifier_match = re.search(r'[^/]+/(\S+)\.fasta', first_line)
    library_name = library_match.group(1) if library_match else "unknown"
    identifier = identifier_match.group(1) if identifier_match else "unknown"

    # Extract position information from the line with "identity"
    identity_line = next((line for line in lines if "identity" in line), None)
    identity_match = re.search(r'(\d+\.\d+)% identity', identity_line) if identity_line else None
    identity_percentage = identity_match.group(1) if identity_match else "NaN"

    # Extract position information from the line containing "identity"
    position_line = next((line for line in lines if "identity" in line), None)
    position_match = re.search(r'(\d+)-(\d+):(\d+)-(\d+)', position_line) if position_line else None
    start_position = position_match.group(3) if position_match else "NaN"
    end_position = position_match.group(4) if position_match else "NaN"

    # Update the NaN counter
    if identity_percentage == "NaN":
        nan_counter += 1

    # Write the result to the output file
    output_line = f"{identifier}\t{library_name}_{start_position}_{end_position}\t{identity_percentage}"

    with open(output_file_name, 'a') as output_file:  # Use 'a' for append mode
        output_file.write(output_line + '\n')  # Add a newline to separate entries

    print(f"Result written to {output_file_name}: {output_line}")

    return nan_counter

def find_and_run_glsearch(file_path):
    failed_count = 0
    nan_counter = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            second_column = columns[1]
            third_column = columns[2]

            # Construct folder paths
            pdb_folder_path = os.path.join("/dsimb/auber/gelly/PROJECTS/PROJECT_PROTEIN_UNITS_ghouzam/wagram_peeling/data/PDBs_Clean", second_column, "Peeling")
            peeling_fasta_path = os.path.join(pdb_folder_path, f"{columns[0]}.fasta")
            prot_sequences_fasta_path = os.path.join("prot_sequences", f"{third_column}.fasta")

            # Check if both fasta files exist
            if os.path.exists(peeling_fasta_path) and os.path.exists(prot_sequences_fasta_path):
                # Construct the output file path
                output_folder_path = "mapping"
                output_file_path = os.path.join(output_folder_path, f"{columns[0]}_{third_column}.txt")

                # Copy peeling fasta file to the mapping directory
                copied_peeling_fasta_path = os.path.join(output_folder_path, f"{columns[0]}.fasta")
                shutil.copy(peeling_fasta_path, copied_peeling_fasta_path)

                # Run glsearch36 command using the copied file
                command = f"glsearch36 -O {output_file_path} {copied_peeling_fasta_path} {prot_sequences_fasta_path}"
                subprocess.run(command, shell=True)
                print(f"Command executed: {command}")

                # Process the glsearch output
                nan_counter = process_glsearch_output(output_file_path, "mapping_summary.txt", nan_counter)

                # Delete the copied file
                os.remove(copied_peeling_fasta_path)
            else:
                print(f"Fasta file not found for {columns[0]} or {third_column}")
                failed_count += 1

    print(f"\nNumber of files not found: {failed_count}")
    print(f"Number of NaN identities: {nan_counter}")

# Specify the path to the input file
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"

# Call the function to find and run glsearch for each line in the input file
find_and_run_glsearch(input_file_path)


## Add ID_Coordinates into PU_nPU_list.txt as a new column ------------------------------------------------------------------------------------
# Read the content of mapping_summary.txt and store the second column in a dictionary
import os
mapping_dict = {}
with open("mapping_summary.txt", "r") as mapping_file:
    for line in mapping_file:
        parts = line.strip().split('\t')
        pdb_name = parts[0]
        mapping_dict[pdb_name] = parts[1]

# Update PU_nPU_list.txt with the new column
input_filename = "/home/auber/merabet/PU_nPU_list.txt"
temp_filename = "/home/auber/merabet/temp_PU_nPU_list.txt"
# Update PU_nPU_list.txt with the new column
with open("/home/auber/merabet/PU_nPU_list.txt", "r") as input_file, open("/home/auber/merabet/temp_PU_nPU_list.txt", "w") as output_file:
    for line in input_file:
        # Split the existing columns
        columns = line.strip().split('\t')
        
        # Extract the fist column (pdb_name)
        pdb_name = columns[0]
        
        # Get the corresponding uniprot value from the dictionary
        uniprot = mapping_dict.get(pdb_name, "NA")
        
        # Insert the new uniprot column as the fourth column
        new_line = '\t'.join([columns[0], columns[1], columns[2], uniprot] + columns[3:])
        
        # Write the modified line to the output file
        output_file.write(new_line + '\n')

# Replace the original file with the modified file
os.replace("/home/auber/merabet/temp_PU_nPU_list.txt", "/home/auber/merabet/PU_nPU_list.txt")


## Remove all the lines where the mapping found zero identity ---------------------------------------------------------------------------------
# Specify the input file name
input_file = "/home/auber/merabet/PU_nPU_list.txt"

# Read the input file and filter lines
with open(input_file, 'r') as infile:
    # Read all lines into a list
    lines = infile.readlines()

# Filter lines based on the fourth column
filtered_lines = [line for line in lines if "NaN_NaN" not in line.split('\t')[3]]

# Write the filtered lines back to the original file
with open(input_file, 'w') as outfile:
    outfile.writelines(filtered_lines)


## Switch between the third and the fourth column ---------------------------------------------------------------------------------------------
file_path = "/home/auber/merabet/PU_nPU_list.txt"

# Read the content of the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line and switch the values in the third and fourth columns
modified_lines = []
for line in lines:
    columns = line.strip().split('\t')
    if len(columns) >= 4:
        columns[2], columns[3] = columns[3], columns[2]
        modified_lines.append('\t'.join(columns) + '\n')
    else:
        # Handle lines with less than four columns (if any)
        modified_lines.append(line)

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.writelines(modified_lines)


## Add 4 "NaN" columns for non-PU -------------------------------------------------------------------------------------------------------------
# Define the input file name
file_name = "/home/auber/merabet/PU_nPU_list.txt"

# Read the content of the file
with open(file_name, 'r') as file:
    lines = file.readlines()

# Modify the content
modified_lines = []
for line in lines:
    # Split the line into columns based on tab delimiter
    columns = line.strip().split('\t')

    # Check if the fifth column is "non_PU"
    if columns[4] == "non_PU":
        # Add four columns with "NaN" values
        columns.extend(["NaN", "NaN", "NaN", "NaN"])

    # Join the modified columns and add the line to the list
    modified_lines.append('\t'.join(columns) + '\n')

# Write the modified content back to the original file
with open(file_name, 'w') as file:
    file.writelines(modified_lines)


## Copy PU_nPU_list.txt as copy_non_filtred_PU_nPU_list.txt -----------------------------------------------------------------------------------
import shutil

source_file = "/home/auber/merabet/PU_nPU_list.txt"
destination_file = "/home/auber/merabet/copy_non_filtred_PU_nPU_list.txt"

shutil.copyfile(source_file, destination_file)


## Find PUs defined as leaders and examples in the same time (mentionned within the same line) ------------------------------------------------
import re
from collections import Counter

file_path = "/dsimb/auber/gelly/PROJECTS/PU_CLASSIFICATIONS/LISTS_CATFS_PU_PROTOTYPE_with_fused_famillies"
output_file_path = "/home/auber/merabet/repeats_in_same_lines.txt"

# Function to find all repeated patterns within the same line
def repeated_patterns_in_line(line):
    patterns = re.findall(r'\b(\w+\_\d+\_\d+\_\d+)\b', line)
    pattern_counter = Counter(patterns)
    repeated_patterns = [pattern for pattern, count in pattern_counter.items() if count >= 2]
    return repeated_patterns

with open(file_path, 'r') as file, open(output_file_path, 'w') as output_file:
    for i, line in enumerate(file):
        # Find repeated patterns within the same line
        repeated_patterns = repeated_patterns_in_line(line)
        
        # Write results to the output file
        for pattern in repeated_patterns:
            output_file.write(f"Pattern: {pattern}\n")
            output_file.write(f"Line Number: {i + 1}\n")
            output_file.write(f"Line: {line.strip()}\n")
            output_file.write("\n")


## Remove duplicate lines ---------------------------------------------------------------------------------------------------------------------
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"

# Read input file and store unique lines in a set
unique_lines = set()
with open(input_file_path, 'r') as input_file:
    for line in input_file:
        unique_lines.add(line.strip())

# Write unique lines back to the original file
with open(input_file_path, 'w') as output_file:
    output_file.write('\n'.join(unique_lines))


## Find PUs having different NanoFold classification (mentionned many times in different lines) -----------------------------------------------
import re
from collections import defaultdict

file_path = "/dsimb/auber/gelly/PROJECTS/PU_CLASSIFICATIONS/LISTS_CATFS_PU_PROTOTYPE_with_fused_famillies"
output_file_path = "/home/auber/merabet/repeats_in_diff_lines.txt"

# Function to find all repeated patterns that occur on different lines
def find_repeated_patterns(patterns_by_line):
    repeated_patterns = {pattern: indices for pattern, indices in patterns_by_line.items() if len(indices) > 1}
    return repeated_patterns

patterns_by_line = defaultdict(list)

with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        # Using regular expression to find patterns
        patterns = re.findall(r'\b(\w+\_\d+\_\d+\_\d+)\b', line)

        if patterns:
            for pattern in set(patterns):  # Ensure uniqueness within a line
                patterns_by_line[pattern].append(i + 1)

# Find all repeated patterns that occur on different lines
repeated_patterns = find_repeated_patterns(patterns_by_line)

# Write the result to the output file
with open(output_file_path, 'w') as output_file:
    for pattern, indices in repeated_patterns.items():
        output_file.write(f"Pattern: {pattern}\n")
        output_file.write(f"Line Numbers: {', '.join(map(str, indices))}\n")
        output_file.write("Lines:\n")
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i + 1 in indices:
                    output_file.write(f"  Line {i + 1}: {line.strip()}\n")
        output_file.write("\n")


## Find overlapping classification at the NanoFold level and correct them by creating new architectures and topologies
file_path = "/home/auber/merabet/PU_nPU_list.txt"
output_file_path = "/home/auber/merabet/overlapping_classifications.txt"

# Dictionary to store nanofold values and associated topology frequencies
nanofold_topology_dict = {}

with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split('\t')
        if len(columns) >= 9:
            nanofold_value = columns[8]
            topology_value = columns[7]
            architecture_value = columns[6]  # Added line to get architecture value

            # Create a unique key for the nanofold within the same line
            key = f"{nanofold_value}"

            if key in nanofold_topology_dict:
                if topology_value in nanofold_topology_dict[key]:
                    nanofold_topology_dict[key][topology_value][architecture_value] += 1  # Updated line
                else:
                    nanofold_topology_dict[key][topology_value] = {architecture_value: 1}  # Updated line
            else:
                nanofold_topology_dict[key] = {topology_value: {architecture_value: 1}}  # Updated line

# Write results to the output file for Nanofold values with at least two different topology values
with open(output_file_path, 'w') as output_file:
    for nanofold_value, topology_dict in nanofold_topology_dict.items():
        if len(topology_dict) >= 2:
            output_file.write(f"Nanofold value with at least two different topology values: {nanofold_value}\n")
            for topology_value, architecture_dict in topology_dict.items():  # Updated line
                for architecture_value, count in architecture_dict.items():  # Updated line
                    output_file.write(f"architecture value: {architecture_value}, topology value: {topology_value}, Nb of PUs: {count}\n")  # Updated line
            most_frequent_topology = max(topology_dict, key=lambda x: sum(topology_dict[x].values()))
            most_frequent_architecture = max(architecture_dict, key=architecture_dict.get)
            output_file.write(f"new architecture value: F{most_frequent_architecture}, new topology value: F{most_frequent_topology}\n\n")  # Updated line

# Process the most frequent combination and update the original file
with open(file_path, 'r') as input_file, open(file_path + '.temp', 'w') as temp_file:
    for line in input_file:
        columns = line.strip().split('\t')
        if len(columns) >= 9:
            nanofold_value = columns[8]
            if nanofold_value in nanofold_topology_dict and len(nanofold_topology_dict[nanofold_value]) >= 2:
                most_frequent_topology = max(nanofold_topology_dict[nanofold_value], key=lambda x: sum(nanofold_topology_dict[nanofold_value][x].values()))  # Updated line

                # Check if the line's nanofold_value matches and apply changes
                if columns[8] == nanofold_value:
                    new_topology_value = f"F{most_frequent_topology}"
                    line = line.replace(columns[7], new_topology_value, 1)

                    # Apply changes to architecture values for all lines with the same nanofold value
                    for architecture_value, count in nanofold_topology_dict[nanofold_value][most_frequent_topology].items():
                        new_architecture_value = f"F{architecture_value}"
                        line = line.replace(columns[6], new_architecture_value, 1)

        temp_file.write(line)

# Rename the temporary file to the original file
import os
os.rename(file_path + '.temp', file_path)


## Filter out the motifs for which the embedding was not done because of the large protein length ---------------------------------------------
# This code should be run after running the code in generate_embedding_ankh.py so that you generate no_embedded_motifs.txt
motifs_file_path = "/home/auber/merabet/no_embedded_motifs.txt"
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"

# Read motifs to be excluded
with open(motifs_file_path, 'r') as motifs_file:
    motifs_to_exclude = set(motif.strip() for motif in motifs_file)

# Process input file and overwrite with modified content
with open(input_file_path, 'r') as input_file, open(input_file_path + ".tmp", 'w') as temp_output_file:
    for line in input_file:
        # Split line into columns using tab as delimiter
        columns = line.strip().split('\t')

        # Check if the motif in the first column is in the set of motifs to exclude
        if columns[0] not in motifs_to_exclude:
            # If not, write the line to the temporary output file
            temp_output_file.write(line)

# Rename the temporary output file to overwrite the original file
import os
os.rename(input_file_path + ".tmp", input_file_path)
