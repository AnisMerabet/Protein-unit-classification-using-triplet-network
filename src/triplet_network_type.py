# Make sure you generate the data list using make_data_list.py, generate embedding using generate_embedding_ankh.py and generate datasets
#using generate_datasets.py before running this code
## Create and train the model -----------------------------------------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_distances
import umap
import numpy as np

# Define the triplet network with GRU layers
class TripletGRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TripletGRUNetwork, self).__init__()
        # Shared GRU layer
        self.shared_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, fc_size).to(device)

    def forward(self, anchor_batch, positive_batch, negative_batch):
        # Get the lengths of sequences in the batch
        anchor_lengths = [tensor.shape[0] for tensor in anchor_batch]
        positive_lengths = [tensor.shape[0] for tensor in positive_batch]
        negative_lengths = [tensor.shape[0] for tensor in negative_batch]
        
        # Find the maximum sequence length for each sequence type in the batch
        max_length = max(max(seq.shape[0] for seq in sequences) for sequences in [anchor_batch, positive_batch, negative_batch])

        # Pad sequences to the same maximum length within a batch
        anchor_batch = pad_sequence([torch.cat([seq, torch.zeros(max_length - seq.shape[0], seq.shape[1])]) for seq in anchor_batch], batch_first=True)
        positive_batch = pad_sequence([torch.cat([seq, torch.zeros(max_length - seq.shape[0], seq.shape[1])]) for seq in positive_batch], batch_first=True)
        negative_batch = pad_sequence([torch.cat([seq, torch.zeros(max_length - seq.shape[0], seq.shape[1])]) for seq in negative_batch], batch_first=True)
        
        # Move to device
        anchor_batch = anchor_batch.to(device)
        positive_batch = positive_batch.to(device)
        negative_batch = negative_batch.to(device)

        # Pack sequences to handle variable lengths
        packed_anchor = pack_padded_sequence(anchor_batch, anchor_lengths, batch_first=True, enforce_sorted=False)
        packed_positive = pack_padded_sequence(positive_batch, positive_lengths, batch_first=True, enforce_sorted=False)
        packed_negative = pack_padded_sequence(negative_batch, negative_lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through the GRU layers
        _, gru1_hidden = self.shared_gru(packed_anchor)
        _, gru2_hidden = self.shared_gru(packed_positive)
        _, gru3_hidden = self.shared_gru(packed_negative)
        
        # Apply fully connected layer with sigmoid activation
        gru1_hidden_fc = self.fc(gru1_hidden)
        gru2_hidden_fc = self.fc(gru2_hidden)
        gru3_hidden_fc = self.fc(gru3_hidden)

        return gru1_hidden_fc, gru2_hidden_fc, gru3_hidden_fc
    
# Define a custom dataset for loading triplets
class TripletDataset(Dataset):
    def __init__(self, file_path, embedding_path):
        # Load triplets from file
        with open(file_path, 'r') as file:
            self.triplets = [line.strip().split('\t') for line in file][:]

        self.embedding_path = embedding_path

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative, anchor_category, anchor_weight = self.triplets[idx]
        
        # Load embeddings
        anchor_embedding = torch.load(f'{self.embedding_path}/{anchor}.pt')
        positive_embedding = torch.load(f'{self.embedding_path}/{positive}.pt')
        negative_embedding = torch.load(f'{self.embedding_path}/{negative}.pt')

        # Remove the first dimension
        anchor_embedding = anchor_embedding.squeeze(0)
        positive_embedding = positive_embedding.squeeze(0)
        negative_embedding = negative_embedding.squeeze(0)

        return anchor_embedding, positive_embedding, negative_embedding, anchor, anchor_category, anchor_weight
        
# Directory to save the models
model_save_dir = '/home/auber/merabet/models'
# File to save epoch-wise evaluation metrics
eval_info_file = '/home/auber/merabet/model_evaluation/eval_type/model_eval_type.txt'

# Define parameters
input_size = 768  # Embedding dimension
hidden_size = 32  # GRU hidden size
fc_size = 16
num_layers = 1
num_epochs = 1000
batch_size = 32

# Define parameters for early stopping
patience = 5  # Number of epochs with no improvement after which training will be stopped
best_validation_loss = float('inf')
counter_no_improvement = 0

# Move the tensors to the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the triplet GRU network
triplet_gru_network = TripletGRUNetwork(input_size, hidden_size, num_layers)

# Move model to device
triplet_gru_network = TripletGRUNetwork(input_size, hidden_size, num_layers).to(device)

# Print the model's structure
print("Model's structure ----------------------------------------")
print(triplet_gru_network)
# Count the number of trainable parameters
num_trainable_params = sum(p.numel() for p in triplet_gru_network.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")
# Count the number of non-trainable parameters
num_non_trainable_params = sum(p.numel() for p in triplet_gru_network.parameters() if not p.requires_grad)
print(f"Number of non-trainable parameters: {num_non_trainable_params}")
print("----------------------------------------------------------")

# Define the optimizer and loss function
optimizer = optim.Adam(triplet_gru_network.parameters(), lr=0.0005)

# Load datasets
train_dataset = TripletDataset('/home/auber/merabet/datasets/train_type.txt', '/home/auber/merabet/PU_nPU_embeddings')

# Load the validation dataset
validation_dataset = TripletDataset('/home/auber/merabet/datasets/validation_type.txt', '/home/auber/merabet/PU_nPU_embeddings')

# Training loop
num_triplets = len(train_dataset)
best_epoch_embeddings = None  # Variable to store embeddings at the best epoch

for epoch in range(num_epochs):
    # Set the model to training mode
    triplet_gru_network.train()
    
    # Create empty lists to accumulate embeddings and other information over batches
    all_anchor_names = []
    all_anchor_categories = []
    all_gru1_hidden = []
    
    for i in range(0, num_triplets, batch_size):
        batch_indices = torch.arange(i, min(i + batch_size, num_triplets))

        # Extract triplet batches using the sampled indices
        triplet_batch = [train_dataset[idx] for idx in batch_indices]
        anchor_batch, positive_batch, negative_batch, anchor_names, anchor_categories, anchor_weights = zip(*triplet_batch)

        # Forward pass through the network
        gru1_hidden, gru2_hidden, gru3_hidden = triplet_gru_network(anchor_batch, positive_batch, negative_batch)

        # Append batch information to the accumulated lists
        all_anchor_names.extend(anchor_names)
        all_anchor_categories.extend(anchor_categories)
        all_gru1_hidden.extend(gru1_hidden[0])
        
        # Define the margin for triplet loss
        margin = 3.0
        
        # Normalize the vectors
        gru1_hidden_normalized = F.normalize(gru1_hidden[0], p=2, dim=1)
        gru2_hidden_normalized = F.normalize(gru2_hidden[0], p=2, dim=1)
        gru3_hidden_normalized = F.normalize(gru3_hidden[0], p=2, dim=1)

        # Apply the cosine similarity function to the hidden states
        similarity_positive = F.cosine_similarity(gru1_hidden_normalized, gru2_hidden_normalized)
        similarity_negative = F.cosine_similarity(gru1_hidden_normalized, gru3_hidden_normalized)

        # Calculate cosine distance
        distance_positive = 1 - similarity_positive
        distance_negative = 1 - similarity_negative

        # Calculate triplet margin loss
        triplet_loss = F.relu(distance_positive - distance_negative + margin)

        # Compute the mean loss over the batch
        loss = torch.mean(triplet_loss)
        
        # Apply category weights to the triplet loss
        #weighted_triplet_loss = triplet_loss * torch.tensor([float(weight) for weight in anchor_weights], device=device)

        # Compute the mean loss over the batch
        #loss = torch.mean(weighted_triplet_loss)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop
    # Set the model to evaluation mode
    triplet_gru_network.eval()
    
    with torch.no_grad():
        validation_loss = 0.0
        num_validation_triplets = len(validation_dataset)

        for i in range(0, num_validation_triplets, batch_size):
            batch_indices = torch.arange(i, min(i + batch_size, num_validation_triplets))

            # Extract triplet batches using the sampled indices
            triplet_batch_val = [validation_dataset[idx] for idx in batch_indices]
            anchor_batch_val, positive_batch_val, negative_batch_val, anchor_names_val, anchor_categories_val, anchor_weights_val = zip(*triplet_batch_val)

            # Forward pass through the network
            gru1_hidden_val, gru2_hidden_val, gru3_hidden_val = triplet_gru_network(anchor_batch_val, positive_batch_val, negative_batch_val)

            # Define the margin for triplet loss
            margin = 3.0

            # Normalize the vectors
            gru1_hidden_val_normalized = F.normalize(gru1_hidden_val[0], p=2, dim=1)
            gru2_hidden_val_normalized = F.normalize(gru2_hidden_val[0], p=2, dim=1)
            gru3_hidden_val_normalized = F.normalize(gru3_hidden_val[0], p=2, dim=1)

            # Apply the cosine similarity function to the hidden states
            similarity_positive_val = F.cosine_similarity(gru1_hidden_val_normalized, gru2_hidden_val_normalized)
            similarity_negative_val = F.cosine_similarity(gru1_hidden_val_normalized, gru3_hidden_val_normalized)
            
            # Calculate cosine distance
            distance_positive = 1 - similarity_positive_val
            distance_negative = 1 - similarity_negative_val

            # Calculate triplet margin loss with category weights
            triplet_loss_val = F.relu(distance_positive - distance_negative + margin)

            # Compute the mean loss over the batch
            loss_val = torch.mean(triplet_loss_val)
            
            # Apply category weights to the triplet loss
            #weighted_triplet_loss_val = triplet_loss_val * torch.tensor([float(weight) for weight in anchor_weights_val], device=device)

            # Compute the mean loss over the batch
            #loss_val = torch.mean(weighted_triplet_loss_val)

            validation_loss += loss_val.item()

        average_validation_loss = validation_loss / (num_validation_triplets // batch_size)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {average_validation_loss:.4f}')

    # Save epoch-wise evaluation metrics
    with open(eval_info_file, 'a') as eval_file:
        eval_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {average_validation_loss:.4f}\n')
    
    # Save model checkpoint
    #model_checkpoint_path = os.path.join(model_save_dir, f'tripletn_type_epoch{epoch + 1}_.h5')
    #torch.save(triplet_gru_network.state_dict(), model_checkpoint_path)
    
    # Save model checkpoint if it's the best so far and save anchor embeddings
    if average_validation_loss < best_validation_loss:
        best_validation_loss = average_validation_loss
        best_epoch = epoch + 1  # Add 1 because epochs are 1-indexed
        best_epoch_embeddings = {
            'epoch': best_epoch,
            'anchor_names': all_anchor_names,
            'anchor_categories': all_anchor_categories,
            'gru1_hidden': all_gru1_hidden
        }
        model_checkpoint_path = os.path.join(model_save_dir, f'tripletn_type_best_model.h5')
        torch.save(triplet_gru_network.state_dict(), model_checkpoint_path)
        counter_no_improvement = 0
    else:
        counter_no_improvement += 1

    # Check for early stopping
    if counter_no_improvement >= patience:
        print(f'Early stopping! No improvement for {patience} epochs.')
        break

# Save embeddings at the best epoch
if best_epoch_embeddings is not None:
    for j, anchor_name in enumerate(best_epoch_embeddings['anchor_names']):
        # Select the embedding of the current anchor from the batch
        anchor_embedding = best_epoch_embeddings['gru1_hidden'][j]

        # Save each anchor embedding with the corresponding name and category
        anchor_category = best_epoch_embeddings['anchor_categories'][j]
        anchor_embedding_path = os.path.join('/home/auber/merabet/model_embeddings/embeddings_type', f'{anchor_name}_{anchor_category}_type.pt')
        torch.save(anchor_embedding, anchor_embedding_path)

# Write the best epoch and best validation loss into model_eval_type.txt
with open(eval_info_file, 'a') as eval_file:
    eval_file.write(f'Best Epoch: {best_epoch}, Best Validation Loss: {best_validation_loss:.4f}\n')
print('Training finished.')

# Model evaluation
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics.pairwise import cosine_distances
import statsmodels.api as sm
import umap
import seaborn as sns
import random

# Define the triplet network with GRU layers
class TripletGRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TripletGRUNetwork, self).__init__()
        # Shared GRU layer
        self.shared_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, fc_size).to(device)

    def forward(self, anchor_batch, positive_batch, negative_batch):
        # Get the lengths of sequences in the batch
        anchor_lengths = [tensor.shape[0] for tensor in anchor_batch]
        positive_lengths = [tensor.shape[0] for tensor in positive_batch]
        negative_lengths = [tensor.shape[0] for tensor in negative_batch]
        
        # Find the maximum sequence length for each sequence type in the batch
        max_length = max(max(seq.shape[0] for seq in sequences) for sequences in [anchor_batch, positive_batch, negative_batch])

        # Pad sequences to the same maximum length within a batch
        anchor_batch = pad_sequence([torch.cat([seq, torch.zeros(max_length - seq.shape[0], seq.shape[1])]) for seq in anchor_batch], batch_first=True)
        positive_batch = pad_sequence([torch.cat([seq, torch.zeros(max_length - seq.shape[0], seq.shape[1])]) for seq in positive_batch], batch_first=True)
        negative_batch = pad_sequence([torch.cat([seq, torch.zeros(max_length - seq.shape[0], seq.shape[1])]) for seq in negative_batch], batch_first=True)
        
        # Move to device
        anchor_batch = anchor_batch.to(device)
        positive_batch = positive_batch.to(device)
        negative_batch = negative_batch.to(device)

        # Pack sequences to handle variable lengths
        packed_anchor = pack_padded_sequence(anchor_batch, anchor_lengths, batch_first=True, enforce_sorted=False)
        packed_positive = pack_padded_sequence(positive_batch, positive_lengths, batch_first=True, enforce_sorted=False)
        packed_negative = pack_padded_sequence(negative_batch, negative_lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through the GRU layers
        _, gru1_hidden = self.shared_gru(packed_anchor)
        _, gru2_hidden = self.shared_gru(packed_positive)
        _, gru3_hidden = self.shared_gru(packed_negative)
        
        # Apply fully connected layer with sigmoid activation
        gru1_hidden_fc = self.fc(gru1_hidden)
        gru2_hidden_fc = self.fc(gru2_hidden)
        gru3_hidden_fc = self.fc(gru3_hidden)

        return gru1_hidden_fc, gru2_hidden_fc, gru3_hidden_fc

# Define a custom dataset for loading triplets
class TripletDataset(Dataset):
    def __init__(self, file_path, embedding_path):
        # Load triplets from file
        with open(file_path, 'r') as file:
            self.triplets = [line.strip().split('\t') for line in file][:]

        self.embedding_path = embedding_path

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative, anchor_category, anchor_weight = self.triplets[idx]
        
        # Load embeddings
        anchor_embedding = torch.load(f'{self.embedding_path}/{anchor}.pt')
        positive_embedding = torch.load(f'{self.embedding_path}/{positive}.pt')
        negative_embedding = torch.load(f'{self.embedding_path}/{negative}.pt')

        # Remove the first dimension
        anchor_embedding = anchor_embedding.squeeze(0)
        positive_embedding = positive_embedding.squeeze(0)
        negative_embedding = negative_embedding.squeeze(0)

        return anchor_embedding, positive_embedding, negative_embedding, anchor, anchor_category, anchor_weight

eval_info_file = '/home/auber/merabet/model_evaluation/eval_type/model_eval_type.txt' # Path to the model evaluation file
model_save_dir = '/home/auber/merabet/models' # Directory to the saved models
embedding_dir = '/home/auber/merabet/model_embeddings/embeddings_type'  # Directory containing embedding files

# Move the tensors to the same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define parameters
input_size = 768  # Embedding dimension
hidden_size = 32  # GRU hidden size
fc_size = 16
num_layers = 1  # Number of GRU layers
batch_size = 32


## Plot PCA, MDS, t-SNE and UMAP for embedding of 50% of anchors saved during model training --------------------------------------------------------
# Set the path to the directory containing the .pt files
pt_files_dir = '/home/auber/merabet/model_embeddings/embeddings_type/'

# List all the .pt files in the directory
all_pt_files = [file for file in os.listdir(pt_files_dir) if file.endswith('.pt')]

# Randomly select 50% of the files
selected_pt_files = random.sample(all_pt_files, k=int(0.5 * len(all_pt_files)))

embeddings = []
categories = []

for selected_pt_file in selected_pt_files:
    # Extract anchor_name and anchor_category from the file name
    anchor_name = "_".join(selected_pt_file.split('_')[:3])
    anchor_category = "_".join(selected_pt_file.split('_')[4:-1])

    # Load the embedding from the selected .pt file
    anchor_embedding = torch.load(os.path.join(pt_files_dir, selected_pt_file))

    # Append the loaded embedding and category
    embeddings.append(anchor_embedding)
    categories.append(anchor_category)

# Convert loaded embeddings and categories to numpy arrays
embeddings_np = torch.stack(embeddings).cpu().detach().numpy()

# Convert string categories to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
categories_numeric = label_encoder.fit_transform(categories)

# Apply PCA to reduce the dimensionality to 2D
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings_np)

# Plot the PCA visualization with colors based on categories
plt.figure(figsize=(10, 8))
for label, category_name in zip(set(categories_numeric), set(categories)):
    indices = categories_numeric == label
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)
    
    # Draw ellipse around each category
    ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                        width=np.std(embeddings_2d[indices, 0]) * 2,
                        height=np.std(embeddings_2d[indices, 1]) * 2,
                        edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
    plt.gca().add_patch(ellipse)

plt.title('PCA Visualization of Anchor Embeddings')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend()
plt.savefig('/home/auber/merabet/model_evaluation/eval_type/PCA_train_type.png')
print("PCA plot saved to /home/auber/merabet/model_evaluation/eval_type/PCA_train_type.png")

# Compute pairwise cosine distances
dissimilarities = cosine_distances(embeddings_np)

# Apply MDS to reduce the dimensionality to 2D
mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed', normalized_stress=False)  # Set normalized_stress to False
embeddings_2d = mds.fit_transform(dissimilarities)

# Plot the MDS visualization with ellipses around each category
plt.figure(figsize=(10, 8))
for label, category_name in zip(set(categories_numeric), set(categories)):
    indices = categories_numeric == label
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)

    # Draw ellipse around each category
    ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                        width=np.std(embeddings_2d[indices, 0]) * 2,
                        height=np.std(embeddings_2d[indices, 1]) * 2,
                        edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
    plt.gca().add_patch(ellipse)

plt.title('MDS Visualization of Anchor Embeddings')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.legend()
plt.savefig('/home/auber/merabet/model_evaluation/eval_type/MDS_train_type.png')
print("MDS plot saved to /home/auber/merabet/model_evaluation/eval_type/MDS_train_type.png")

# Apply t-SNE to reduce the dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_np)

# Plot the t-SNE visualization with colors based on categories
plt.figure(figsize=(10, 8))
for label, category_name in zip(set(categories_numeric), set(categories)):
    indices = categories_numeric == label
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)

    # Draw ellipse around each category
    ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                        width=np.std(embeddings_2d[indices, 0]) * 2,
                        height=np.std(embeddings_2d[indices, 1]) * 2,
                        edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
    plt.gca().add_patch(ellipse)

plt.title('t-SNE Visualization of Anchor Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.savefig('/home/auber/merabet/model_evaluation/eval_type/tSNE_train_type.png')
print("t-SNE plot saved to /home/auber/merabet/model_evaluation/eval_type/tSNE_train_type.png")

# Apply UMAP to reduce the dimensionality to 2D
umap_model = umap.UMAP(n_components=2, n_jobs=-1, random_state=None)  # Do not set random_state
embeddings_2d = umap_model.fit_transform(embeddings_np)

# Plot the UMAP visualization with colors based on categories
plt.figure(figsize=(10, 8))
for label, category_name in zip(set(categories_numeric), set(categories)):
    indices = categories_numeric == label
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)
    
    # Draw ellipse around each category
    ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                        width=np.std(embeddings_2d[indices, 0]) * 2,
                        height=np.std(embeddings_2d[indices, 1]) * 2,
                        edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
    plt.gca().add_patch(ellipse)
    
plt.title('UMAP Visualization of Anchor Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend()
plt.savefig('/home/auber/merabet/model_evaluation/eval_type/UMAP_train_type.png')
print("UMAP plot saved to /home/auber/merabet/model_evaluation/eval_type/UMAP_train_type.png")


## Evaluate the model on the test dataset -----------------------------------------------------------------------------------------------------
# Load the best model
best_model_path = os.path.join(model_save_dir, 'tripletn_type_best_model.h5')
best_triplet_gru_network = TripletGRUNetwork(input_size, hidden_size, num_layers).to(device)
best_triplet_gru_network.load_state_dict(torch.load(best_model_path))
best_triplet_gru_network.eval()

# Load the test dataset
test_dataset = TripletDataset('/home/auber/merabet/datasets/test_type.txt', '/home/auber/merabet/PU_nPU_embeddings')

# Initialize counters for the first closest file category and the 5 first closest file categories
total_true_count_first_closest = 0
total_pred_count_first_closest = 0
category_counts_first_closest = {}

total_true_count_5_closest = 0
total_pred_count_5_closest = 0
category_counts_5_closest = {}

# Initialize lists to store true positive rate (tpr) and false positive rate (fpr)
tpr_list = []
fpr_list = []

# Lists to store thresholds and Youden's index values
thresholds_list = []
youden_index_list = []

# Evaluate the model on the test dataset
best_triplet_gru_network.eval()
with torch.no_grad():
    test_loss = 0.0
    num_test_triplets = len(test_dataset)

    distance_positive_list = []
    distance_negative_list = []
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # Create empty lists to accumulate embeddings and other information over batches
    all_anchor_categories = []
    all_gru1_hidden = []

    for i in range(0, num_test_triplets, batch_size):
        batch_indices_test = torch.arange(i, min(i + batch_size, num_test_triplets))

        # Extract triplet batches using the sampled indices
        triplet_batch_test = [test_dataset[idx] for idx in batch_indices_test]
        anchor_batch_test, positive_batch_test, negative_batch_test, anchor_names_test, anchor_categories_test, anchor_weights_test  = zip(*triplet_batch_test)

        # Forward pass through the network
        gru1_hidden_test, gru2_hidden_test, gru3_hidden_test = best_triplet_gru_network(anchor_batch_test, positive_batch_test, negative_batch_test)

        # Append batch information to the accumulated lists
        all_anchor_categories.extend(anchor_categories_test)
        all_gru1_hidden.extend(gru1_hidden_test[0])
        
        # Define the margin for triplet loss
        margin = 3.0

        # Normalize the vectors
        gru1_hidden_test_normalized = F.normalize(gru1_hidden_test[0], p=2, dim=1)
        gru2_hidden_test_normalized = F.normalize(gru2_hidden_test[0], p=2, dim=1)
        gru3_hidden_test_normalized = F.normalize(gru3_hidden_test[0], p=2, dim=1)

        # Apply the cosine similarity function to the hidden states
        similarity_positive_test = F.cosine_similarity(gru1_hidden_test_normalized, gru2_hidden_test_normalized)
        similarity_negative_test = F.cosine_similarity(gru1_hidden_test_normalized, gru3_hidden_test_normalized)

        # Calculate cosine distance
        distance_positive = 1 - similarity_positive_test
        distance_negative = 1 - similarity_negative_test
        
        # Append to lists for plotting
        distance_positive_list.extend(distance_positive.cpu().numpy())
        distance_negative_list.extend(distance_negative.cpu().numpy())
        
        # Calculate triplet margin loss with category weights
        triplet_loss_test = F.relu(distance_positive - distance_negative + margin)

        # Compute the mean loss over the batch
        loss_test = torch.mean(triplet_loss_test)
        
        # Apply category weights to the triplet loss
        #weighted_triplet_loss_test = triplet_loss_test * torch.tensor([float(weight) for weight in anchor_weights_test], device=device)

        # Compute the mean loss over the batch
        #loss_test = torch.mean(weighted_triplet_loss_test)

        test_loss += loss_test.item()
        
        # Calculate and print the file with the lowest cosine distance for each anchor embedding
        for j, anchor_embedding_test_normalized in enumerate(gru1_hidden_test_normalized):
            anchor_embedding_test_normalized = gru1_hidden_test_normalized[j]

            # Load all files from the embedding directory
            embedding_files = os.listdir(embedding_dir)

            # Initialize variables for minimum distance and corresponding file names
            min_distance_first_closest = float('inf')
            min_distance_file_first_closest = None

            min_distances_5_closest = [float('inf')] * 5
            min_distance_files_5_closest = [None] * 5

            # Calculate cosine distances and find the minimum distances
            for embedding_file in embedding_files:
                # Load the embedding from the file
                loaded_embedding = torch.load(os.path.join(embedding_dir, embedding_file))
                # Normalize the vector
                loaded_embedding_normalized = F.normalize(loaded_embedding.view(1, -1), p=2, dim=1)

                # Calculate cosine distance
                cosine_similarity = F.cosine_similarity(anchor_embedding_test_normalized, loaded_embedding_normalized)
                distance = 1 - cosine_similarity

                # Update the minimum distance for the first closest category
                if distance < min_distance_first_closest:
                    min_distance_first_closest = distance
                    min_distance_file_first_closest = embedding_file

                # Update the minimum distances for the 5 closest categories
                for i in range(5):
                    if distance < min_distances_5_closest[i]:
                        min_distances_5_closest[i] = distance
                        min_distance_files_5_closest[i] = embedding_file
                        break

            # Extract category from the closest file name for the first closest category
            closest_file_category_first_closest = "_".join(min_distance_file_first_closest.split('_')[4:-1])

            # Extract categories from the closest file names for the 5 closest categories
            closest_file_categories_5_closest = ["_".join(file.split('_')[4:-1]) for file in min_distance_files_5_closest]

            # Extract category from the anchor file name
            anchor_category = anchor_categories_test[j]

            # Check if the categories are identical for the first closest category
            is_identical_category_first_closest = anchor_category == closest_file_category_first_closest

            # Check if the anchor category is equal to at least one of the closest file categories for the 5 closest categories
            is_identical_category_5_closest = anchor_category in closest_file_categories_5_closest

            # Update total counters for the first closest category
            total_true_count_first_closest += is_identical_category_first_closest
            total_pred_count_first_closest += 1

            # Update category-specific counters for the first closest category
            if anchor_category not in category_counts_first_closest:
                category_counts_first_closest[anchor_category] = {"true_count": 0, "pred_count": 0}

            category_counts_first_closest[anchor_category]["true_count"] += is_identical_category_first_closest
            category_counts_first_closest[anchor_category]["pred_count"] += 1

            # Update total counters for the 5 closest categories
            total_true_count_5_closest += is_identical_category_5_closest
            total_pred_count_5_closest += 1

            # Update category-specific counters for the 5 closest categories
            if anchor_category not in category_counts_5_closest:
                category_counts_5_closest[anchor_category] = {"true_count": 0, "pred_count": 0}

            category_counts_5_closest[anchor_category]["true_count"] += is_identical_category_5_closest
            category_counts_5_closest[anchor_category]["pred_count"] += 1
    
    # Specify the embeddings and categories
    embeddings = all_gru1_hidden
    categories = all_anchor_categories
    
    # Convert embeddings and categories to numpy arrays
    embeddings_np = torch.stack(embeddings).cpu().detach().numpy()
    
    # Convert string categories to numeric labels using LabelEncoder
    label_encoder = LabelEncoder()
    categories_numeric = label_encoder.fit_transform(categories)

    # Apply PCA to reduce the dimensionality to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_np)

    # Plot the PCA visualization with colors based on categories
    plt.figure(figsize=(10, 8))
    for label, category_name in zip(set(categories_numeric), set(categories)):
        indices = categories_numeric == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)
        
        # Draw ellipse around each category
        ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                          width=np.std(embeddings_2d[indices, 0]) * 2,
                          height=np.std(embeddings_2d[indices, 1]) * 2,
                          edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
        plt.gca().add_patch(ellipse)

    plt.title('PCA Visualization of Anchor Embeddings')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend()
    plt.savefig('/home/auber/merabet/model_evaluation/eval_type/PCA_test_type.png')
    print("PCA plot saved to /home/auber/merabet/model_evaluation/eval_type/PCA_test_type.png")

    # Compute pairwise cosine distances
    dissimilarities = cosine_distances(embeddings_np)

    # Apply MDS to reduce the dimensionality to 2D
    mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed', normalized_stress=False)  # Set normalized_stress to False
    embeddings_2d = mds.fit_transform(dissimilarities)

    # Plot the MDS visualization with ellipses around each category
    plt.figure(figsize=(10, 8))
    for label, category_name in zip(set(categories_numeric), set(categories)):
        indices = categories_numeric == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)

        # Draw ellipse around each category
        ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                          width=np.std(embeddings_2d[indices, 0]) * 2,
                          height=np.std(embeddings_2d[indices, 1]) * 2,
                          edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
        plt.gca().add_patch(ellipse)

    plt.title('MDS Visualization of Anchor Embeddings')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.legend()
    plt.savefig('/home/auber/merabet/model_evaluation/eval_type/MDS_test_type.png')
    print("MDS plot saved to /home/auber/merabet/model_evaluation/eval_type/MDS_test_type.png")

    # Apply t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    # Plot the t-SNE visualization with colors based on categories
    plt.figure(figsize=(10, 8))
    for label, category_name in zip(set(categories_numeric), set(categories)):
        indices = categories_numeric == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)
    
        # Draw ellipse around each category
        ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                          width=np.std(embeddings_2d[indices, 0]) * 2,
                          height=np.std(embeddings_2d[indices, 1]) * 2,
                          edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
        plt.gca().add_patch(ellipse)

    plt.title('t-SNE Visualization of Anchor Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig('/home/auber/merabet/model_evaluation/eval_type/tSNE_test_type.png')
    print("t-SNE plot saved to /home/auber/merabet/model_evaluation/eval_type/tSNE_test_type.png")

    # Apply UMAP to reduce the dimensionality to 2D
    umap_model = umap.UMAP(n_components=2, n_jobs=-1, random_state=None)  # Do not set random_state
    embeddings_2d = umap_model.fit_transform(embeddings_np)

    # Plot the UMAP visualization with colors based on categories
    plt.figure(figsize=(10, 8))
    for label, category_name in zip(set(categories_numeric), set(categories)):
        indices = categories_numeric == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category_name)
        
        # Draw ellipse around each category
        ellipse = Ellipse(xy=(np.mean(embeddings_2d[indices, 0]), np.mean(embeddings_2d[indices, 1])),
                          width=np.std(embeddings_2d[indices, 0]) * 2,
                          height=np.std(embeddings_2d[indices, 1]) * 2,
                          edgecolor=f'C{label}', facecolor='none')  # Use color of the category for the ellipse
        plt.gca().add_patch(ellipse)
        
    plt.title('UMAP Visualization of Anchor Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.savefig('/home/auber/merabet/model_evaluation/eval_type/UMAP_test_type.png')
    print("UMAP plot saved to /home/auber/merabet/model_evaluation/eval_type/UMAP_test_type.png")

   # Plot the distance distributions
    plt.figure(figsize=(10, 6))
    plt.hist(distance_positive_list, bins=50, alpha=0.5, label='Positive Distances', color='blue')
    plt.hist(distance_negative_list, bins=50, alpha=0.5, label='Negative Distances', color='red')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Positive and Negative Distances')
    plt.legend()
    plt.savefig('/home/auber/merabet/model_evaluation/eval_type/distances_test_type.png')
    print("Distances Histogram Plot saved to /home/auber/merabet/model_evaluation/eval_type/distances_test_type.png")
    plt.clf()  # or 
    plt.close()
    
    # Logistic Regression Plot
    # Combine positive and negative distances
    all_distances = np.concatenate([distance_positive_list, distance_negative_list])
    # Create labels (1 for positive, 0 for negative)
    labels = np.concatenate([np.ones_like(distance_positive_list), np.zeros_like(distance_negative_list)])

    # Fit GLM with binomial family
    X = sm.add_constant(all_distances)
    model = sm.GLM(labels, X, family=sm.families.Binomial())
    result = model.fit()

    # Predict probabilities
    distances_for_prediction = np.linspace(all_distances.min(), all_distances.max(), 300)
    X_pred = sm.add_constant(distances_for_prediction)
    probabilities = result.predict(X_pred)

    # Plot Logistic Regression
    plt.figure(figsize=(10, 6))
    plt.scatter(all_distances, labels, alpha=0.5, color='gray', label='Data Points')

    # Plot GLM line
    plt.plot(distances_for_prediction, probabilities, color='blue', label='Generalized Linear Model')

    plt.xlabel('Network output')
    plt.ylabel('Probability')
    plt.title('Generalized Linear Model Fit')
    plt.legend()
    plt.savefig('/home/auber/merabet/model_evaluation/eval_type/logistic_regression_type.png')
    print("Logistic Regression Plot saved to /home/auber/merabet/model_evaluation/eval_type/logistic_regression_type.png")

    # Iterate through different threshold values to compute ROC curve
    thresholds = np.arange(0, 2.01, 0.01)
    for threshold in thresholds:
        # Calculate True Positive, False Positive, True Negative, False Negative
        true_positives_to_thld = np.sum(np.array(distance_positive_list) < threshold)
        false_positives_to_thld = np.sum(np.array(distance_negative_list) < threshold)
        true_negatives_to_thld = np.sum(np.array(distance_negative_list) >= threshold)
        false_negatives_to_thld = np.sum(np.array(distance_positive_list) >= threshold)

        # Calculate True Positive Rate (tpr) and False Positive Rate (fpr)
        tpr = true_positives_to_thld / (true_positives_to_thld + false_negatives_to_thld)
        fpr = false_positives_to_thld / (false_positives_to_thld + true_negatives_to_thld)

        # Append to lists
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
        # Calculate Youden's index and append to lists
        youden_index = tpr + (1 - fpr) - 1
        thresholds_list.append(threshold)
        youden_index_list.append(youden_index)
        
    # Find the threshold corresponding to the maximum Youden's index
    optimal_threshold = thresholds_list[np.argmax(youden_index_list)]
    optimal_youden_index = np.max(youden_index_list)
    
    # Calculate Area Under the Curve (AUC)
    roc_auc = auc(fpr_list, tpr_list)
    print("ROC AUC:", roc_auc)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.scatter([1-optimal_threshold], [tpr_list[thresholds_list.index(optimal_threshold)]], c='red', marker='o', label=f'Optimal Youden\'s Index (Threshold = {optimal_threshold:.2f})')
    plt.plot([1-optimal_threshold, 1-optimal_threshold], [0, tpr_list[thresholds_list.index(optimal_threshold)]], linestyle='--', color='red')  # Vertical line for optimal threshold
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # Save ROC curve plot
    roc_curve_save_path = '/home/auber/merabet/model_evaluation/eval_type/roc_curve_type.png'
    plt.savefig(roc_curve_save_path)
    print("ROC Curve saved to", roc_curve_save_path)
    print("Optimal Youden's Index:", optimal_youden_index)

    # Calculate Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(np.array([1]*len(distance_positive_list) + [0]*len(distance_negative_list)),
                                                            -np.concatenate([distance_positive_list, distance_negative_list]))

    # Calculate PR AUC
    pr_auc = auc(recall, precision)
    print("Precision-Recall AUC:", pr_auc)

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    # Save Precision-Recall curve plot
    pr_curve_save_path = '/home/auber/merabet/model_evaluation/eval_type/pr_curve_type.png'
    plt.savefig(pr_curve_save_path)
    print("Precision-Recall Curve saved to", pr_curve_save_path)

    # Calculate true positives, false positives and false negatives based on the threshold
    for distance_positive in distance_positive_list:
        true_positives += (distance_positive < optimal_threshold).sum().item()
        false_negatives += (distance_positive >= optimal_threshold).sum().item()

    for distance_negative in distance_negative_list:
        false_positives += (distance_negative < optimal_threshold).sum().item()
        true_negatives += (distance_negative >= optimal_threshold).sum().item()

    # Calculate confusion matrix
    conf_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('/home/auber/merabet/model_evaluation/eval_type/confusion_matrix_type.png')
    print("Confusion Matrix saved to /home/auber/merabet/model_evaluation/eval_type/confusion_matrix_type.png")
    
    # Calculate Precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    print(f'Precision: {precision:.4f}')

    # Calculate Recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    print(f'Recall: {recall:.4f}')

    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Print F1-score
    print(f'F1-score: {f1_score:.4f}')

    # Calculate Matthews correlation coefficient (MCC)
    mcc_denominator = np.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))
    mcc = (true_positives * true_negatives - false_positives * false_negatives) / mcc_denominator if mcc_denominator != 0 else 0
    # Print MCC
    print(f'MCC: {mcc:.4f}')

    # Calculate Balance Error Rate (BER)
    ber = 0.5 * ((false_positives / (true_negatives + false_positives)) + (false_negatives / (true_positives + false_negatives)))
    # Print BER
    print(f'Balance Error Rate (BER): {ber:.4f}')

    # Define the cases to iterate over
    cases = [
        {"name": "first_closest", "counters": (total_true_count_first_closest, total_pred_count_first_closest, category_counts_first_closest)},
        {"name": "5_closest", "counters": (total_true_count_5_closest, total_pred_count_5_closest, category_counts_5_closest)}
    ]

    # Initialize a dictionary to store all results
    all_results = {}

    for case in cases:
        # Unpack counters
        total_true_count, total_pred_count, category_counts = case["counters"]

        # Initialize dictionaries to store results specific to each category
        category_results = {}
        for category in category_counts.keys():
            category_results[category] = {
                "true_count": 0,
                "pred_count": 0,
                "ratio": 0
            }

        # Calculate and store the ratio for each category
        for category, counts in category_counts.items():
            ratio = counts["true_count"] / counts["pred_count"] if counts["pred_count"] > 0 else 0
            category_results[category]["true_count"] = counts["true_count"]
            category_results[category]["pred_count"] = counts["pred_count"]
            category_results[category]["ratio"] = ratio

        # Sort categories based on the ratio and category name
        sorted_categories = sorted(category_results.keys(), key=lambda x: (category_results[x]["ratio"], x.lower()))

        # Reverse the sorted list
        sorted_categories.reverse()

        # Extract ratios and categories for plotting
        ratios = [category_results[category]["ratio"] for category in sorted_categories]
        categories = sorted_categories

        # Store results in the all_results dictionary
        all_results[case["name"]] = {
            "total_true_count": total_true_count,
            "total_pred_count": total_pred_count,
            "total_ratio": total_true_count / total_pred_count if total_pred_count > 0 else 0,
            "category_results": category_results
        }

        # Plotting bar plot for ratios according to categories
        plt.figure(figsize=(8, 8))
        plt.bar(categories, ratios, color='blue')
        plt.xlabel('Categories')
        plt.ylabel('Ratios')
        plt.title(f'Category Ratios - {case["name"].replace("_", " ").capitalize()}')
        plt.xticks(rotation=45, ha='right')
        bar_plot_category_path = f'/home/auber/merabet/model_evaluation/eval_type/barplot_category_type_{case["name"]}.png'
        plt.savefig(bar_plot_category_path)
        print(f"Category bar plot for {case['name']} saved to", bar_plot_category_path)
        plt.close()  # Close the plot to avoid overlapping plots
    
    # Save results to a file
    output_file_path = '/home/auber/merabet/model_evaluation/eval_type/categories_eval_type.txt'
    with open(output_file_path, 'w') as output_file:
        # Save Optimal Threshold
        output_file.write(f'Optimal Threshold = {optimal_threshold:.4f}\n')
        
        # Save Precision
        output_file.write(f'Precision (PRE) = {precision:.4f}\n')
        
        # Save Recall
        output_file.write(f'Recall (SPE) = {recall:.4f}\n')

        # Save F1-score
        output_file.write(f'F1-score: {f1_score:.4f}\n')

        # Save MCC
        output_file.write(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}\n')

        # Save BER
        output_file.write(f'Balance Error Rate (BER): {ber:.4f}\n\n')

        # Save results to a file
        for case_name, results in all_results.items():
            output_file.write(f'Case: {case_name.capitalize()}\n')
            output_file.write(f'Total True Predictions: {results["total_true_count"]}, Total Predictions: {results["total_pred_count"]}, Total Ratio: {results["total_ratio"]:.4f}\n')
            output_file.write('Category-specific Ratios:\n')
            # Sort categories based on the ratio and category name
            sorted_categories = sorted(results["category_results"].keys(), key=lambda x: (results["category_results"][x]["ratio"], x.lower()))
            # Reverse the sorted list
            sorted_categories.reverse()
            for category in sorted_categories:
                result = results["category_results"][category]
                output_file.write(f'    Category: {category}, True Predictions: {result["true_count"]}, Total Predictions: {result["pred_count"]}, Ratio: {result["ratio"]:.4f}\n')
        # Print a confirmation message
        print(f'Results saved to: {output_file_path}')

    average_test_loss = test_loss / (num_test_triplets // batch_size)
print(f'Test Loss: {average_test_loss:.4f}')
# Append the test loss to the model_eval_type.txt file
with open(eval_info_file, 'a') as eval_file:
    eval_file.write(f'Test Loss: {average_test_loss:.4f}\n')
print(f'Test Loss saved to {eval_info_file}')

## Plot Loss over epochs ----------------------------------------------------------------------------------------------------------------------
# Read the model evaluation file
with open(eval_info_file, 'r') as eval_file:
    lines = eval_file.readlines()

# Parse the information from the file
train_losses = []
val_losses = []
best_epoch = 0
best_val_loss = float('inf')
test_loss = 0.0

for line in lines:
    if ', Loss:' in line:
        # Splitting line using ',' and ':' as separators
        epoch_info, loss_info, val_loss_info = line.split(',')
        
        # Extracting loss values
        _, train_loss_str = loss_info.split(':')
        _, val_loss_str = val_loss_info.split(':')
        
        train_losses.append(float(train_loss_str.strip()))
        val_losses.append(float(val_loss_str.strip()))
    elif 'Best Epoch:' in line:
        # Using regular expression to extract values
        match = re.search(r'(\d+),.*?(\d+\.\d+)', line)
        if match:
            best_epoch, best_val_loss = map(float, match.groups())
    elif 'Test Loss:' in line:
        # Splitting line using ':' as separator
        _, test_loss_str = line.split(':')
        test_loss = float(test_loss_str.strip())

# Plot the losses
last_epoch = len(train_losses) + 1
plt.figure(figsize=(10, 6))
plt.plot(range(1, last_epoch), train_losses, label='Train Loss')
plt.plot(range(1, last_epoch), val_losses, label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({int(best_epoch)})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Loss Over Epochs')

# Define the path to save the plot
save_path = '/home/auber/merabet/model_evaluation/eval_type/Loss_plot_Type.png'

# Save the plot
plt.savefig(save_path, format='png', bbox_inches='tight')
print("Loss Plot saved to", save_path)