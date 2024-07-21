## Count the number of motifs------------------------------------------------------------------------------------------------------------------
file_path = '/home/auber/merabet/PU_nPU_list.txt'

# Open the file and count the lines
with open(file_path, 'r') as file:
    motif_count = sum(1 for line in file)

# Print the count
print(f'Number of motifs: {motif_count}')


## Count the number of unique protein IDs------------------------------------------------------------------------------------------------------
file_path = '/home/auber/merabet/PU_nPU_list.txt'

# Read the file and extract unique values from the fourth column
unique_protein_ids = set()
with open(file_path, 'r') as file:
    for line in file:
        columns = line.strip().split('\t')
        protein_id = columns[3]
        unique_protein_ids.add(protein_id)

# Print the number of unique protein IDs
print("Number of unique protein IDs:", len(unique_protein_ids))


## Calculate motif lengths and save into a txt file--------------------------------------------------------------------------------------------
input_file_path = '/home/auber/merabet/PU_nPU_list.txt'
output_file_path = '/home/auber/merabet/data_exploration/motif_lengths.txt'

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        columns = line.strip().split('\t')
        motif_name = columns[2]
        start, end = map(int, motif_name.split('_')[1:])
        length = end - start + 1
        output_file.write(f"{motif_name}\t{length}\n")

print("Motif lengths file created successfully.")


## Calculate motif lengths for PU and non_PU separetely and save into a txt file---------------------------------------------------------------
input_file_path = '/home/auber/merabet/PU_nPU_list.txt'

# Output file paths
pu_output_file_path = '/home/auber/merabet/data_exploration/PU_lengths.txt'
non_pu_output_file_path = '/home/auber/merabet/data_exploration/non_PU_lengths.txt'

with open(input_file_path, 'r') as input_file, \
     open(pu_output_file_path, 'w') as pu_output_file, \
     open(non_pu_output_file_path, 'w') as non_pu_output_file:

    for line in input_file:
        columns = line.strip().split('\t')
        motif_name = columns[2]
        start, end = map(int, motif_name.split('_')[1:])
        length = end - start + 1

        # Check the value in column 5 and write to the appropriate file
        if columns[4] == 'PU':
            pu_output_file.write(f"{motif_name}\t{length}\n")
        elif columns[4] == 'non_PU':
            non_pu_output_file.write(f"{motif_name}\t{length}\n")

print("Motif lengths files created successfully.")


## Create a bar chart for amino acid composition for PUs and non_PUs and perform t-test to check the difference--------------------------------
import os
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Function to read fasta file and count amino acids
def count_amino_acids(fasta_file):
    amino_acid_counter = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0,
                          'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequence = str(record.seq)
        for aa in amino_acid_counter:
            amino_acid_counter[aa] += sequence.count(aa)

    total_count = sum(amino_acid_counter.values())
    amino_acid_percentage = {aa: count / total_count * 100 for aa, count in amino_acid_counter.items()}

    return amino_acid_percentage

# Function to perform t-test for each amino acid type
def perform_t_test(non_pu_data, pu_data):
    p_values = {}

    for aa in non_pu_data:
        t_stat, p_value = ttest_ind(non_pu_data[aa], pu_data[aa], equal_var=False)
        p_values[aa] = p_value

    return p_values

# Function to process the input file and generate the bar chart
def generate_amino_acid_chart(input_file, output_file):
    data = pd.read_csv(input_file, sep='\t', header=None, names=['motif_name', 'ID_PDB', 'Uniprot_mapping', 'ID_Uniprot', 'Type', 'Class', 'Architecture', 'Topology', 'NanoFold'])

    amino_acid_total_percentage_non_pu = {'A': [], 'R': [], 'N': [], 'D': [], 'C': [], 'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
                                           'L': [], 'K': [], 'M': [], 'F': [], 'P': [], 'S': [], 'T': [], 'W': [], 'Y': [], 'V': []}

    amino_acid_total_percentage_pu = {'A': [], 'R': [], 'N': [], 'D': [], 'C': [], 'Q': [], 'E': [], 'G': [], 'H': [], 'I': [],
                                       'L': [], 'K': [], 'M': [], 'F': [], 'P': [], 'S': [], 'T': [], 'W': [], 'Y': [], 'V': []}

    for index, row in data.iterrows():
        pdb_name = row['ID_PDB']
        fasta_path = f'/dsimb/auber/gelly/PROJECTS/PROJECT_PROTEIN_UNITS_ghouzam/wagram_peeling/data/PDBs_Clean/{pdb_name}/Peeling/{row["motif_name"]}.fasta'

        if os.path.exists(fasta_path):
            amino_acid_percentage = count_amino_acids(fasta_path)

            for aa, percentage in amino_acid_percentage.items():
                if row['Type'] == 'non_PU':
                    amino_acid_total_percentage_non_pu[aa].append(percentage)
                elif row['Type'] == 'PU':
                    amino_acid_total_percentage_pu[aa].append(percentage)

    total_samples_non_pu = len(data[data['Type'] == 'non_PU'])
    total_samples_pu = len(data[data['Type'] == 'PU'])

    average_amino_acid_percentage_non_pu = {aa: sum(percentages) / total_samples_non_pu for aa, percentages in amino_acid_total_percentage_non_pu.items()}
    average_amino_acid_percentage_pu = {aa: sum(percentages) / total_samples_pu for aa, percentages in amino_acid_total_percentage_pu.items()}

    # Perform t-test
    p_values = perform_t_test(amino_acid_total_percentage_non_pu, amino_acid_total_percentage_pu)

    # Apply Bonferroni correction
    bonferroni_factor = len(p_values)
    adjusted_p_values = {aa: p_value * bonferroni_factor for aa, p_value in p_values.items()}

    # Print significant results
    significant_aa = [aa for aa, p_value in adjusted_p_values.items() if p_value < 0.05]

    for aa in significant_aa:
        print(f"Significant difference for amino acid {aa}: adjusted p-value = {adjusted_p_values[aa]}")

    # Plotting the bar chart
    plt.bar(average_amino_acid_percentage_non_pu.keys(), average_amino_acid_percentage_non_pu.values(), label='non_PU', alpha=0.5)
    plt.bar(average_amino_acid_percentage_pu.keys(), average_amino_acid_percentage_pu.values(), label='PU', alpha=0.5)

    # Add stars above significant bars
    for aa in significant_aa:
        index = list(average_amino_acid_percentage_non_pu.keys()).index(aa)
        plt.annotate('*', xy=(index, max(average_amino_acid_percentage_non_pu[aa], average_amino_acid_percentage_pu[aa]) + 0.5),
                     ha='center', va='bottom', fontsize=12)

    plt.xlabel('Amino Acid')
    plt.ylabel('Percentage')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

# Run the script with the provided file paths
input_file_path = '/home/auber/merabet/PU_nPU_list.txt'
output_file_path = '/home/auber/merabet/data_exploration/PU_non_PU_AA_frequency.png'
print("Positive results of independent t-test each amino acid composition between PU and non_PU sequences")
generate_amino_acid_chart(input_file_path, output_file_path)

print(f"PU and non_PU amino acid composition bar chart saved to {output_file_path}")


## Create a bar chart for amino acid category composition for PUs and non_PUs and perform t-test to check the difference--------------------------------
import os
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Function to read fasta file and count amino acids
def count_amino_acids(fasta_file):
    amino_acid_counter = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0,
                          'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}

    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequence = str(record.seq)
        for aa in amino_acid_counter:
            amino_acid_counter[aa] += sequence.count(aa)

    total_count = sum(amino_acid_counter.values())
    amino_acid_percentage = {aa: count / total_count * 100 for aa, count in amino_acid_counter.items()}

    return amino_acid_percentage

# Function to categorize amino acids
def categorize_amino_acids():
    categories = {
        'acidic': ['D', 'E'],
        'acyclic': ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'I', 'L', 'K', 'M', 'S', 'T', 'V'],
        'aliphatic': ['A', 'G', 'I', 'L', 'V'],
        'aromatic': ['H', 'F', 'W', 'Y'],
        'basic': ['R', 'H', 'K'],
        'buried': ['A', 'C', 'I', 'L', 'M', 'F', 'W', 'V'],
        'charged': ['R', 'D', 'E', 'H', 'K'],
        'cyclic': ['H', 'F', 'P', 'W', 'Y'],
        'hydrophobic': ['A', 'G', 'I', 'L', 'M', 'F', 'P', 'W', 'Y', 'V'],
        'large': ['R', 'E', 'Q', 'H', 'I', 'L', 'K', 'M', 'F', 'W', 'Y'],
        'medium': ['N', 'D', 'C', 'P', 'T', 'V'],
        'negative': ['D', 'E'],
        'neutral': ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
        'polar': ['R', 'N', 'D', 'C', 'E', 'Q', 'H', 'K', 'S', 'T'],
        'positive': ['R', 'H', 'K'],
        'small': ['A', 'G', 'S'],
        'surface': ['R', 'N', 'D', 'E', 'Q', 'G', 'H', 'K', 'P', 'S', 'T', 'Y']
    }
    return categories

# Function to categorize amino acids in a sequence
def categorize_sequence_amino_acids(amino_acid_percentage, categories):
    category_counts = {category: 0 for category in categories}

    for aa, percentage in amino_acid_percentage.items():
        for category, amino_acids in categories.items():
            if aa in amino_acids:
                category_counts[category] += percentage

    return category_counts


# Function to perform t-test for each category
def perform_t_test_for_categories(non_pu_data, pu_data):
    p_values = {}

    for category in non_pu_data:
        t_stat, p_value = ttest_ind(non_pu_data[category], pu_data[category], equal_var=False)
        p_values[category] = p_value

    return p_values

# Function to process the input file and generate the bar chart for categories
def generate_category_chart(input_file, output_file):
    data = pd.read_csv(input_file, sep='\t', header=None, names=['motif_name', 'ID_PDB', 'Uniprot_mapping', 'ID_Uniprot', 'Type', 'Class', 'Architecture', 'Topology', 'NanoFold'])

    categories = categorize_amino_acids()

    category_total_percentage_non_pu = {category: [] for category in categories}
    category_total_percentage_pu = {category: [] for category in categories}

    for index, row in data.iterrows():
        pdb_name = row['ID_PDB']
        fasta_path = f'/dsimb/auber/gelly/PROJECTS/PROJECT_PROTEIN_UNITS_ghouzam/wagram_peeling/data/PDBs_Clean/{pdb_name}/Peeling/{row["motif_name"]}.fasta'

        if os.path.exists(fasta_path):
            amino_acid_percentage = count_amino_acids(fasta_path)
            category_counts = categorize_sequence_amino_acids(amino_acid_percentage, categories)

            for category, count in category_counts.items():
                if row['Type'] == 'non_PU':
                    category_total_percentage_non_pu[category].append(count)
                elif row['Type'] == 'PU':
                    category_total_percentage_pu[category].append(count)

    total_samples_non_pu = len(data[data['Type'] == 'non_PU'])
    total_samples_pu = len(data[data['Type'] == 'PU'])

    average_category_percentage_non_pu = {category: sum(counts) / total_samples_non_pu for category, counts in category_total_percentage_non_pu.items()}
    average_category_percentage_pu = {category: sum(counts) / total_samples_pu for category, counts in category_total_percentage_pu.items()}

    # Perform t-test for each category
    p_values = perform_t_test_for_categories(category_total_percentage_non_pu, category_total_percentage_pu)

    # Apply Bonferroni correction
    bonferroni_factor = len(p_values)
    adjusted_p_values = {category: p_value * bonferroni_factor for category, p_value in p_values.items()}

    # Print significant results
    significant_categories = [category for category, p_value in adjusted_p_values.items() if p_value < 0.05]

    for category in significant_categories:
        print(f"Significant difference for category {category}: adjusted p-value = {adjusted_p_values[category]}")

    # Plotting the bar chart with diagonal category labels
    fig, ax = plt.subplots()

    # Calculate the absolute difference in percentage between PU and non_PU for each category
    percentage_diff = {category: (average_category_percentage_pu[category] - average_category_percentage_non_pu[category]) for category in categories}

    # Sort categories based on the absolute difference
    sorted_categories = sorted(percentage_diff, key=percentage_diff.get, reverse=True)

    # Plot bars for sorted categories
    ax.bar([category for category in sorted_categories], [average_category_percentage_non_pu[category] for category in sorted_categories], label='non_PU', alpha=0.5)
    ax.bar([category for category in sorted_categories], [average_category_percentage_pu[category] for category in sorted_categories], label='PU', alpha=0.5)

    # Add stars above significant bars
    for category in significant_categories:
        index = sorted_categories.index(category)
        ax.annotate('*', xy=(index, max(average_category_percentage_non_pu[category], average_category_percentage_pu[category]) + 0.5),
                    ha='center', va='bottom', fontsize=12)

    plt.xlabel('Amino Acid Category')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')  # Rotate category labels diagonally
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent clipping of rotated labels
    plt.savefig(output_file)
    plt.close()

# Run the script with the provided file paths
input_file_path = '/home/auber/merabet/PU_nPU_list.txt'
output_file_path = '/home/auber/merabet/data_exploration/PU_non_PU_AA_category_frequency.png'
print("Positive results of independent t-test for each amino acid category between PU and non_PU sequences")
generate_category_chart(input_file_path, output_file_path)

print(f"PU and non_PU amino acid category composition bar chart saved to {output_file_path}")


## Calculate the number of PU and non_PU and save it into a txt file---------------------------------------------------------------------------
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"
output_file_path = "/home/auber/merabet/data_exploration/PU_nPU_frequency.txt"

# Read the input file and count occurrences of "PU" and "non_PU"
pu_count = 0
non_pu_count = 0

with open(input_file_path, 'r') as input_file:
    for line in input_file:
        columns = line.strip().split('\t')
        if len(columns) >= 5:
            if columns[4] == "PU":
                pu_count += 1
            elif columns[4] == "non_PU":
                non_pu_count += 1

# Write the results to the output file
with open(output_file_path, 'w') as output_file:
    output_file.write("PU\t{}\n".format(pu_count))
    output_file.write("non_PU\t{}\n".format(non_pu_count))

print("PU and non_PU counts written to", output_file_path)


## Create a bar chart for the frequency of PU and non_PU---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/PU_nPU_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
categories = []
values = []
for line in lines:
    category, frequency = line.strip().split()
    categories.append(category)
    values.append(int(frequency))

# Calculate percentages
total = sum(values)
percentages = [(value / total) * 100 for value in values]

# Create a bar chart
fig, ax = plt.subplots()
bars = ax.bar(categories, values)

# Display values on top of the bars
for bar, percentage in zip(bars, percentages):
    height = bar.get_height()
    ax.annotate(f"{height}\n({percentage:.2f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Set labels and title
ax.set_xlabel("Motif Type")
ax.set_ylabel("Frequency")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/PU_nPU_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

print(f"PU/non_PU bar chart saved to {output_file_path}")


## Create a bar chart for the frequency of motifs by bines-------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/motif_lengths.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
motif_types = []
lengths = []
for line in lines:
    motif_type, length = line.strip().split("\t")
    motif_types.append(motif_type)
    lengths.append(int(length))

# Define bin classes
bin_classes = ["0-20", "20-40", "40-60", "60-80", "80-100"]

# Assign each length to the corresponding bin class
class_lengths = [0] * len(bin_classes)
for length, motif_type in zip(lengths, motif_types):
    if length <= 20:
        class_lengths[0] += 1
    elif length <= 40:
        class_lengths[1] += 1
    elif length <= 60:
        class_lengths[2] += 1
    elif length <= 80:
        class_lengths[3] += 1
    else:
        class_lengths[4] += 1

# Create a bar chart
fig, ax = plt.subplots()
bars = ax.bar(bin_classes, class_lengths)

# Display values on top of the bars
for bar, length in zip(bars, class_lengths):
    height = bar.get_height()
    ax.annotate(f"{length}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Set labels and title
ax.set_xlabel("Nb of amino-acids")
ax.set_ylabel("Number of Motifs")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/bins_motifs_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

print(f"Motif bins bar chart saved to {output_file_path}")


## Create a bar chart for the percentages of PU and non_PU by bines----------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Function to read data from a file and extract PU or non_PU data
def read_data(file_path):
    with open(file_path, "r") as input_file:
        lines = input_file.readlines()

    data = []
    lengths = []
    for line in lines:
        entry, length = line.strip().split("\t")
        data.append(entry)
        lengths.append(int(length))

    return data, lengths

# Input file paths
pu_file_path = "/home/auber/merabet/data_exploration/PU_lengths.txt"
non_pu_file_path = "/home/auber/merabet/data_exploration/non_PU_lengths.txt"

# Read data for PU and non_PU
PUs, pu_lengths = read_data(pu_file_path)
non_PUs, non_pu_lengths = read_data(non_pu_file_path)

# Define bin classes
bin_classes = ["0-20", "20-40", "40-60", "60-80", "80-100"]
num_bins = len(bin_classes)

# Initialize data for PU and non_PU
pu_class_lengths = [0] * num_bins
non_pu_class_lengths = [0] * num_bins

# Assign each length to the corresponding bin class for PU
for length, PU in zip(pu_lengths, PUs):
    for i in range(num_bins):
        if length <= (i + 1) * 20:
            pu_class_lengths[i] += 1
            break

# Assign each length to the corresponding bin class for non_PU
for length, non_PU in zip(non_pu_lengths, non_PUs):
    for i in range(num_bins):
        if length <= (i + 1) * 20:
            non_pu_class_lengths[i] += 1
            break

# Calculate percentages for PU and non_PU
total_pu_lengths = sum(pu_class_lengths)
total_non_pu_lengths = sum(non_pu_class_lengths)

pu_percentages = [length / total_pu_lengths * 100 for length in pu_class_lengths]
non_pu_percentages = [length / total_non_pu_lengths * 100 for length in non_pu_class_lengths]

# Create a bar chart with overlapping bars for PU and non_PU
fig, ax = plt.subplots()

width = 0.35  # Width of the bars
bin_positions = range(num_bins)

pu_bars = ax.bar(bin_positions, pu_percentages, width, label='PU')
non_pu_bars = ax.bar([pos + width for pos in bin_positions], non_pu_percentages, width, label='non_PU')

# Display percentage values on top of the bars
for bars, percentages in zip([pu_bars, non_pu_bars], [pu_percentages, non_pu_percentages]):
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f"{percentage:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Set labels and title
ax.set_xlabel("Nb of amino-acids")
ax.set_ylabel("Percentage")
ax.set_xticks([pos + width / 2 for pos in bin_positions])
ax.set_xticklabels(bin_classes)
ax.legend()

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/bins_PU_non_PU_percentage.png"
plt.savefig(output_file_path, bbox_inches='tight')

print(f"Combined bins bar chart saved to {output_file_path}")

# Display the chart
plt.show()


## Calculate the frequency of each class and save it into a txt file---------------------------------------------------------------------------
import os
from collections import Counter

# Input and output file paths
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"
output_file_path = "/home/auber/merabet/data_exploration/class_frequency.txt"

# Read the input file and extract the sixth column
class_column_index = 5
class_frequencies = Counter()

with open(input_file_path, "r") as input_file:
    for line in input_file:
        columns = line.strip().split("\t")
        if len(columns) > class_column_index and columns[class_column_index] != "NaN":
            class_type = columns[class_column_index]
            class_frequencies[class_type] += 1

# Write the frequencies to the output file
with open(output_file_path, "w") as output_file:
    for class_type, frequency in class_frequencies.items():
        output_file.write(f"{class_type}\t{frequency}\n")

num_classes = len(class_frequencies)
print(f"Class frequencies written to {output_file_path}")
print(f"Number of classes: {num_classes}")


## Create a bar chart for the frequency of classes---------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/class_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
categories = []
values = []
for line in lines:
    category, frequency = line.strip().split()
    categories.append(category)
    values.append(int(frequency))

# Calculate percentages
total = sum(values)
percentages = [(value / total) * 100 for value in values]

# Create a bar chart
fig, ax = plt.subplots()
bars = ax.bar(categories, values)

# Display values on top of the bars
for bar, percentage in zip(bars, percentages):
    height = bar.get_height()
    ax.annotate(f"{height}\n({percentage:.2f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Set labels and title
ax.set_xlabel("Class Type")
ax.set_ylabel("Frequency")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/class_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

print(f"Class bar chart saved to {output_file_path}")


## Calculate the frequency of each architecture and save it into a txt file--------------------------------------------------------------------
import os
from collections import Counter

# Input and output file paths
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"
output_file_path = "/home/auber/merabet/data_exploration/architecture_frequency.txt"

# Read the input file and extract the seventh column
architecture_column_index = 6
architecture_frequencies = Counter()

with open(input_file_path, "r") as input_file:
    for line in input_file:
        columns = line.strip().split("\t")
        if len(columns) > architecture_column_index and columns[architecture_column_index] != "NaN":
            architecture_type = columns[architecture_column_index]
            architecture_frequencies[architecture_type] += 1

# Write the frequencies to the output file
with open(output_file_path, "w") as output_file:
    for architecture_type, frequency in architecture_frequencies.items():
        output_file.write(f"{architecture_type}\t{frequency}\n")

num_architectures = len(architecture_frequencies)
print(f"Architecture frequencies written to {output_file_path}")
print(f"Number of architectures: {num_architectures}")


## Create a bar chart for the frequency of architectures---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/architecture_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
architectures = []
frequencies = []
for line in lines:
    architecture, frequency = line.strip().split()
    architectures.append(architecture)
    frequencies.append(int(frequency))

# Sort data by frequency in descending order
sorted_data = sorted(zip(architectures, frequencies), key=lambda x: x[1], reverse=True)
sorted_architectures, sorted_frequencies = zip(*sorted_data)

# Create a bar chart with vertically rotated x-axis labels
fig, ax = plt.subplots()
bars = ax.bar(sorted_architectures, sorted_frequencies)
ax.set_xticklabels(sorted_architectures, rotation=90, fontsize='small')

# Set labels and title
ax.set_xlabel("Architecture Types")
ax.set_ylabel("Frequency")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/architecture_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

print(f"Architecture bar chart saved to {output_file_path}")


## Create a bar chart for the frequency of architectures by bines------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/architecture_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
architecture_types = []
frequencies = []
for line in lines:
    architecture_type, frequency = line.strip().split()
    architecture_types.append(architecture_type)
    frequencies.append(int(frequency))

# Define bin classes
bin_classes = ["0-50", "50-100", "100-150", "150-200", "200-18000"]

# Assign each frequency to the corresponding bin class
class_frequencies = [0] * len(bin_classes)
for frequency, architecture_type in zip(frequencies, architecture_types):
    if frequency <= 50:
        class_frequencies[0] += 1
    elif frequency <= 100:
        class_frequencies[1] += 1
    elif frequency <= 150:
        class_frequencies[2] += 1
    elif frequency <= 200:
        class_frequencies[3] += 1
    else:
        class_frequencies[4] += 1

# Create a bar chart
fig, ax = plt.subplots()
bars = ax.bar(bin_classes, class_frequencies)

# Display values on top of the bars
for bar, frequency in zip(bars, class_frequencies):
    height = bar.get_height()
    ax.annotate(f"{frequency}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Set labels and title
ax.set_xlabel("Frequency")
ax.set_ylabel("Number of Architecture Types")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/bins_architecture_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

# Show the chart
plt.show()

print(f"Architecture bins bar chart saved to {output_file_path}")


## Calculate the frequency of each topology and save it into a txt file------------------------------------------------------------------------
import os
from collections import Counter

# Input and output file paths
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"
output_file_path = "/home/auber/merabet/data_exploration/topology_frequency.txt"

# Read the input file and extract the eighth column
topology_column_index = 7
topology_frequencies = Counter()

with open(input_file_path, "r") as input_file:
    for line in input_file:
        columns = line.strip().split("\t")
        if len(columns) > topology_column_index and columns[topology_column_index] != "NaN":
            topology_type = columns[topology_column_index]
            topology_frequencies[topology_type] += 1

# Write the frequencies to the output file
with open(output_file_path, "w") as output_file:
    for topology_type, frequency in topology_frequencies.items():
        output_file.write(f"{topology_type}\t{frequency}\n")

num_topologies = len(topology_frequencies)
print(f"Topology frequencies written to {output_file_path}")
print(f"Number of topologies: {num_topologies}")


## Create a bar chart for the frequency of topologies------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/topology_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
topologies = []
frequencies = []
for line in lines:
    topology, frequency = line.strip().split()
    topologies.append(topology)
    frequencies.append(int(frequency))

# Sort data in decreasing order of frequency
sorted_indices = sorted(range(len(frequencies)), key=lambda k: frequencies[k], reverse=True)
topologies = [topologies[i] for i in sorted_indices]
frequencies = [frequencies[i] for i in sorted_indices]

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(range(len(topologies)), frequencies)

# Remove x-axis labels
ax.set_xticks([])

# Set labels and title
ax.set_xlabel("Topology Type")
ax.set_ylabel("Frequency")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/topology_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

print(f"Topology bar chart saved to {output_file_path}")


## Create a bar chart for the frequency of topology by bines-----------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/topology_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
topology_types = []
frequencies = []
for line in lines:
    topology_type, frequency = line.strip().split()
    topology_types.append(topology_type)
    frequencies.append(int(frequency))

# Define bin classes
bin_classes = ["0-10", "10-20", "20-30", "30-40", "40-14000"]

# Assign each frequency to the corresponding bin class
class_frequencies = [0] * len(bin_classes)
for frequency, topology_type in zip(frequencies, topology_types):
    if frequency <= 10:
        class_frequencies[0] += 1
    elif frequency <= 20:
        class_frequencies[1] += 1
    elif frequency <= 30:
        class_frequencies[2] += 1
    elif frequency <= 40:
        class_frequencies[3] += 1
    else:
        class_frequencies[4] += 1

# Create a bar chart
fig, ax = plt.subplots()
bars = ax.bar(bin_classes, class_frequencies)

# Display values on top of the bars
for bar, frequency in zip(bars, class_frequencies):
    height = bar.get_height()
    ax.annotate(f"{frequency}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Set labels and title
ax.set_xlabel("Frequency")
ax.set_ylabel("Number of Topology Types")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/bins_topology_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

# Show the chart
plt.show()

print(f"Topology bins bar chart saved to {output_file_path}")


## Calculate the frequency of each NanoFold and save it into a txt file------------------------------------------------------------------------
import os
from collections import Counter

# Input and output file paths
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"
output_file_path = "/home/auber/merabet/data_exploration/nanofold_frequency.txt"

# Read the input file and extract the ninth column
nanofold_column_index = 8
nanofold_frequencies = Counter()

with open(input_file_path, "r") as input_file:
    for line in input_file:
        columns = line.strip().split("\t")
        if len(columns) > nanofold_column_index and columns[nanofold_column_index] != "NaN":
            nanofold_type = columns[nanofold_column_index]
            nanofold_frequencies[nanofold_type] += 1

# Write the frequencies to the output file
with open(output_file_path, "w") as output_file:
    for nanofold_type, frequency in nanofold_frequencies.items():
        output_file.write(f"{nanofold_type}\t{frequency}\n")

num_nanofolds = len(nanofold_frequencies)
print(f"NanoFold frequencies written to {output_file_path}")
print(f"Number of NanoFolds: {num_nanofolds}")


## Create a bar chart for the frequency of NanoFolds-------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/nanofold_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
nanofolds = []
frequencies = []
for line in lines:
    nanofold, frequency = line.strip().split()
    nanofolds.append(nanofold)
    frequencies.append(int(frequency))

# Sort data in decreasing order of frequency
sorted_indices = sorted(range(len(frequencies)), key=lambda k: frequencies[k], reverse=True)
nanofolds = [nanofolds[i] for i in sorted_indices]
frequencies = [frequencies[i] for i in sorted_indices]

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(range(len(nanofolds)), frequencies)

# Remove x-axis labels
ax.set_xticks([])

# Set labels and title
ax.set_xlabel("NanoFold Type")
ax.set_ylabel("Frequency")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/nanofold_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

print(f"NanoFold bar chart saved to {output_file_path}")


## Create a bar chart for the frequency of NanoFolds by bines----------------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Input file path
input_file_path = "/home/auber/merabet/data_exploration/nanofold_frequency.txt"

# Read data from the input file
with open(input_file_path, "r") as input_file:
    lines = input_file.readlines()

# Extract data
nanofold_types = []
frequencies = []
for line in lines:
    nanofold_type, frequency = line.strip().split()
    nanofold_types.append(nanofold_type)
    frequencies.append(int(frequency))

# Define bin classes
bin_classes = ["0-5", "5-10", "10-15", "15-20", "20-1200"]

# Assign each frequency to the corresponding bin class
class_frequencies = [0] * len(bin_classes)
for frequency, nanofold_type in zip(frequencies, nanofold_types):
    if frequency <= 5:
        class_frequencies[0] += 1
    elif frequency <= 10:
        class_frequencies[1] += 1
    elif frequency <= 15:
        class_frequencies[2] += 1
    elif frequency <= 20:
        class_frequencies[3] += 1
    else:
        class_frequencies[4] += 1

# Create a bar chart
fig, ax = plt.subplots()
bars = ax.bar(bin_classes, class_frequencies)

# Display values on top of the bars
for bar, frequency in zip(bars, class_frequencies):
    height = bar.get_height()
    ax.annotate(f"{frequency}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Set labels and title
ax.set_xlabel("Frequency")
ax.set_ylabel("Number of NanoFold Types")

# Save the chart as a PNG file
output_file_path = "/home/auber/merabet/data_exploration/bins_nanofold_frequency.png"
plt.savefig(output_file_path, bbox_inches='tight')

# Show the chart
plt.show()

print(f"NanoFold bins bar chart saved to {output_file_path}")