import os
import pandas as pd
import numpy as np
import random

# Set the path to the input file
input_file_path = "/home/auber/merabet/PU_nPU_list.txt"

# Read the data from the input file
df = pd.read_csv(input_file_path, sep='\t', header=None, names=['motif', 'pdb_id', 'uniprot_mapping', 'uniprot_id', 'type', 'class', 'architecture', 'topology', 'nanofold'])

# Set the seed for reproducibility
np.random.seed(42)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into train, validation, and test sets
train_size = int(0.7 * len(df))
valid_size = int(0.15 * len(df))
test_size = len(df) - train_size - valid_size

train_set = df[:train_size]
valid_set = df[train_size:train_size+valid_size]
test_set = df[train_size+valid_size:]

def calculate_category_weights(num_motifs):
    total_motifs = sum(num_motifs)
    num_categories = len(num_motifs)
    weights = [(total_motifs / (num_categories * count)) for count in num_motifs]
    # Normalize weights in a way that it ranges from 1 to 5
    normalized_weights = [1 + 4 * (weight / max(weights)) for weight in weights]
    normalized_weights = [round(weight, 1) for weight in normalized_weights]
    return normalized_weights

def generate_triplets(column_name, df, train_set, valid_set, test_set, output_directory):

    # Exclude NaN values in the specified column
    df = df.dropna(subset=[column_name])

    # Filter out categories with only one motif
    #column_counts = df[column_name].value_counts()
    #single_motif_categories = column_counts[column_counts == 1].index
    #df = df[~df[column_name].isin(single_motif_categories)]

    # Identify unique categories
    #categories = df[column_name].unique()

    # Calculate category frequencies
    category_counts = df[column_name].value_counts()
    categories = category_counts.index.tolist()
    num_motifs = category_counts.tolist()

    # Calculate category weights
    category_weights = calculate_category_weights(num_motifs)

    # Function to create triplets for train and validation datasets
    def create_triplets_train_val(dataset):
        triplets = []
        # Calculate category frequencies
        category_counts_dataset = dataset[column_name].value_counts()
        categories_dataset = category_counts_dataset.index.tolist()
        num_motifs_dataset = category_counts_dataset.tolist()

        # Identify the category with the greatest number of motifs
        max_motifs_category = categories_dataset[np.argmax(num_motifs_dataset)]
        max_motifs_count = np.max(num_motifs_dataset)

        anchor_rows_by_category = dataset.groupby(column_name)

        for category, anchor_rows in anchor_rows_by_category:
            if category != max_motifs_category:
                triplets_category = []
                
                # If the category does not match max_motifs_category, run through the loop once
                while len(triplets_category) < max_motifs_count:
                    for index, anchor_row in anchor_rows.iterrows():
                        positive_candidates = dataset[dataset[column_name] == anchor_row[column_name]]
                        positive_candidates = positive_candidates[positive_candidates['motif'] != anchor_row['motif']]

                        if len(positive_candidates) == 0:
                            continue

                        positive_example = positive_candidates.sample(1).iloc[0]

                        # Select negative example from a different category
                        negative_candidates = dataset[dataset[column_name] != anchor_row[column_name]]
                        negative_example = negative_candidates.sample(1).iloc[0]
                    
                        #print(len(negative_candidates))
                        negative_example = negative_candidates.sample(1).iloc[0]

                        anchor_category_weight = category_weights[categories.index(anchor_row[column_name])]

                        triplets_category.append([anchor_row['motif'], positive_example['motif'], negative_example['motif'], anchor_row[column_name], anchor_category_weight])
                        triplets.append([anchor_row['motif'], positive_example['motif'], negative_example['motif'], anchor_row[column_name], anchor_category_weight])

                        # Check the condition after each triplet is appended
                        if len(triplets_category) >= max_motifs_count:
                            break
            else:
                # If the category matches max_motifs_category, repeat the loop until max_motifs_count is reached
                for index, anchor_row in anchor_rows.iterrows():
                    positive_candidates = dataset[dataset[column_name] == anchor_row[column_name]]
                    positive_candidates = positive_candidates[positive_candidates['motif'] != anchor_row['motif']]

                    if len(positive_candidates) == 0:
                        continue

                    positive_example = positive_candidates.sample(1).iloc[0]

                    negative_candidates_main = dataset[dataset[column_name] != anchor_row[column_name]]
                    negative_example_main = negative_candidates_main.sample(1).iloc[0]

                    anchor_category_weight = category_weights[categories.index(anchor_row[column_name])]

                    triplets.append([anchor_row['motif'], positive_example['motif'], negative_example_main['motif'], anchor_row[column_name], anchor_category_weight])

        # Shuffling the triplets list
        random.shuffle(triplets)
        
        return triplets
    
    # Function to create triplets for test dataset
    def create_triplets_test(dataset):
        triplets = []
        for index, anchor_row in dataset.iterrows():
            # Select positive example from the same category
            positive_candidates = dataset[dataset[column_name] == anchor_row[column_name]]
            positive_candidates = positive_candidates[positive_candidates['motif'] != anchor_row['motif']]

            # Skip if there are no positive candidates
            if len(positive_candidates) == 0:
                continue

            positive_example = positive_candidates.sample(1).iloc[0]

            # Select negative example from a different category
            negative_candidates = dataset[dataset[column_name] != anchor_row[column_name]]
            negative_example = negative_candidates.sample(1).iloc[0]

            # Get the weight of the category for the anchor motif
            anchor_category_weight = category_weights[categories.index(anchor_row[column_name])]

            triplets.append([anchor_row['motif'], positive_example['motif'], negative_example['motif'], anchor_row[column_name], anchor_category_weight])

        # Shuffling the triplets list
        random.shuffle(triplets)
        
        return triplets

    # Create triplets for each dataset
    train_triplets = create_triplets_train_val(train_set)
    valid_triplets = create_triplets_train_val(valid_set)
    test_triplets = create_triplets_test(test_set)

    # Function to save triplets to a file
    def save_triplets(triplets, file_path):
        with open(file_path, 'w') as file:
            for triplet in triplets:
                file.write('\t'.join(map(str, triplet)) + '\n')

    # Save triplets to files
    save_triplets(train_triplets, os.path.join(output_directory, f'train_{column_name.lower()}.txt'))
    save_triplets(valid_triplets, os.path.join(output_directory, f'validation_{column_name.lower()}.txt'))
    save_triplets(test_triplets, os.path.join(output_directory, f'test_{column_name.lower()}.txt'))

    print(f"Triplets generation for '{column_name}' completed.")

# Set the path to the output directory
output_directory = "/home/auber/merabet/datasets"

# Generate triplets for each column
generate_triplets('type', df, train_set, valid_set, test_set, output_directory)
generate_triplets('class', df, train_set, valid_set, test_set, output_directory)
generate_triplets('architecture', df, train_set, valid_set, test_set, output_directory)
generate_triplets('topology', df, train_set, valid_set, test_set, output_directory)
generate_triplets('nanofold', df, train_set, valid_set, test_set, output_directory)