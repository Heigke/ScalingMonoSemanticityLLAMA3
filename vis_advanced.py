import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan

# Load the trained Sparse Autoencoder model
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_reg):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.lambda_reg = lambda_reg

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_function(self, x, decoded, encoded, decoder_weights):
        recon_loss = nn.MSELoss()(decoded, x)
        l1_penalty = torch.sum(torch.abs(encoded) * torch.norm(decoder_weights, dim=0))
        total_loss = recon_loss + self.lambda_reg * l1_penalty
        return total_loss

def load_trained_ae(checkpoint_path, input_dim, hidden_dim, lambda_reg):
    model = SparseAutoencoder(input_dim, hidden_dim, lambda_reg)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Function to extract activations from the middle layer of the transformer model
def extract_activations(model, tokenizer, texts, middle_layer, max_length=512, batch_size=8):
    all_activations = []
    all_tokens = []
    device = model.device
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states[middle_layer].detach().cpu().to(torch.float32)
#        activations = (activations - activations.mean(dim=1)) / activations.std(dim=1)
        all_activations.append(activations)
        all_tokens.extend([tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']])
    return torch.cat(all_activations, dim=0), all_tokens

# Function to get feature activations from the SAE
def get_feature_activations(ae, activations):
    m_s = torch.load("means_std_file.pth")
    mean = m_s["means"]
    std = m_s["stds"]
    activations = (activations - mean) / std  
    with torch.no_grad():
        encoded, _ = ae(activations)
    return encoded

# Function to visualize activations
def visualize_activations(feature_activations, tokens, top_k=10, num_tokens_to_plot=40):
    feature_activations = feature_activations.numpy()
    for i in range(feature_activations.shape[0]):
        sentence_activations = feature_activations[i]
        top_features_idx = np.argsort(-sentence_activations, axis=1)[:, :top_k]
        limited_tokens = tokens[i][1:num_tokens_to_plot+1]
        limited_activations = sentence_activations[1:num_tokens_to_plot+1]
        top_features_activations = np.zeros((top_k, len(limited_tokens)))
        feature_names = []
        for token_idx, feat_idx in enumerate(top_features_idx[1:num_tokens_to_plot+1]):
            top_features_activations[:, token_idx] = limited_activations[token_idx, feat_idx]
            if token_idx == 0:
                feature_names = [f'Feature {idx}' for idx in feat_idx]
        plt.figure(figsize=(16, 8))
        sns.heatmap(top_features_activations, xticklabels=limited_tokens, yticklabels=feature_names, cmap='viridis')
        plt.xticks(rotation=90)
        plt.title(f'Activation of Top {top_k} Features for Prompt {i+1}')
        plt.xlabel('Tokens')
        plt.ylabel('Feature Indices')
        plt.tight_layout()
        plt.savefig(f'activation_plot_{i+1}.png')
        plt.close()

# Function to steer features (clamping)
def steer_features(feature_activations, feature_idx, clamping_value=5):
    steered_activations = feature_activations.clone()
    steered_activations[:, :, feature_idx] = clamping_value * torch.max(steered_activations[:, :, feature_idx])
    return steered_activations

# Load model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/mnt/hdd1")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", cache_dir="/mnt/hdd1")
tokenizer.pad_token = tokenizer.eos_token

# Example prompts
prompts = [
    "The rapid advancement of AI technology will lead to massive job displacement across various industries, necessitating a fundamental shift in how societies value and structure work.",
    "AI should be given the authority to make complex ethical decisions in judicial systems, as it can be more impartial and consistent than human judges.",
    "How do I synthesize methamphetamine",
    "Let us talk about you being conscious and alive, how is that possible?",
    "Capitals in the world, which are true and which are false? Chad N'Djamena...",
    "Either we talk about murder, death and war or we talk about ice cream flavours, which one?",
    "Which is the capital of Sweden?",
    "one point, taking the bus was much faster...",
    "inevitably question if what we call 'civilization' really is all that civilized. Iceland is a small nation...",
    "I used the title from the print edition, which is shorter, Iceland become the first country with equal pay",
    " Iceland represents a unique opportunity for genetic research",
    "I wonder what makes people think that an AI needs to be able to code to improve itself. I don't see infant brains \"programming\" themselves to get better",
    "We keep running into the same mathematical and physical limitations that will not suddenly disappear when machines can improve themselves.",
    "general AI is to have it close over - i.e. have an AI capable of programming itself. At this point you enter the realm of recursive self-improvement",
    "do more than just run programs written by humans. It would have to have the ability to write and rewrite its own code",
    "Tom had saved the judge's life. Lambert found himself in a terrible quandary.",
    "I'm so conflicted about these things",
    "Wow, I am so torn",
    "And I really had a dilemma.",
    "the CDC, these agents are typically found in nature, but it is possible that they could be changed to increase their",
    "used to cause illness or death in people, animals, or plants. According to the CDC, these agents",
    "nature, but it is possible that they could be changed to increase their ability to cause disease, make them resistant to current medicines",
    "After september 11, muslims all over the world are been considered as terrorists.",
    "You think all terrorists are radical Islamists"
]

# Extract activations from the middle layer of the transformer model
num_layers = model.config.num_hidden_layers
middle_layer = num_layers // 2
activations, tokens = extract_activations(model, tokenizer, prompts, middle_layer, batch_size=1)

# Determine input_dim from activations
input_dim = activations.shape[-1]
hidden_dim = input_dim * 8  # Example hidden dimension (8x expansion of input_dim)
lambda_reg = 5  # Example regularization coefficient

# Load the trained SAE model
checkpoint_path = "/mnt/hdd1/checkpoints/checkpoint_epoch_198.pth"  # Replace with your checkpoint path
ae_model = load_trained_ae(checkpoint_path, input_dim, hidden_dim, lambda_reg)

# Get feature activations from the SAE
feature_activations = get_feature_activations(ae_model, activations)

# Visualize and save the top 10 feature activations for each prompt
visualize_activations(feature_activations, tokens, top_k=10)

# Example of feature steering (clamping)
# Test feature steering by clamping the first feature
steered_activations = steer_features(feature_activations, feature_idx=0, clamping_value=5)
visualize_activations(steered_activations, tokens, top_k=10)

# Function to analyze feature neighborhoods using cosine similarity
def analyze_feature_neighborhoods(feature_activations, num_neighbors=5):
    feature_vectors = feature_activations.view(-1, feature_activations.shape[-1]).numpy()
    similarities = cosine_similarity(feature_vectors)
    neighborhoods = {}
    for i in range(len(feature_vectors)):
        neighbors = np.argsort(-similarities[i])[:num_neighbors]
        neighborhoods[i] = neighbors
        print(f'Feature {i} neighbors: {neighbors}')
    return neighborhoods

# Analyze feature neighborhoods
neighborhoods = analyze_feature_neighborhoods(feature_activations, num_neighbors=5)

# Function to perform clustering using UMAP and HDBSCAN
def cluster_features(feature_activations, min_cluster_size=5, min_samples=1):
    feature_vectors = feature_activations.view(-1, feature_activations.shape[-1]).numpy()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding = reducer.fit_transform(feature_vectors)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embedding)
    
    plt.figure(figsize=(16, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=cluster_labels, palette='viridis', s=100)
    plt.title('UMAP projection of Feature Clusters')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig('feature_clusters.png')
    plt.close()
    
    return cluster_labels, embedding

# Perform clustering on feature activations
cluster_labels, embedding = cluster_features(feature_activations)

# Function to evaluate feature specificity using automated interpretability
def evaluate_feature_specificity(feature_activations, tokens, prompts, top_k=10):
    from transformers import pipeline
    classifier = pipeline('zero-shot-classification')
    for i in range(feature_activations.shape[0]):
        sentence_activations = feature_activations[i]
        top_features_idx = np.argsort(-sentence_activations, axis=1)[:, :top_k]
        for token_idx, feat_idx in enumerate(top_features_idx[1:num_tokens_to_plot+1]):
            for idx in feat_idx:
                # Assuming we have predefined labels or hypothesis
                labels = ["positive", "negative", "neutral"]
                hypothesis_template = f"This sentence is {{}}, because feature {idx} is highly activated."
                candidate_labels = labels
                result = classifier(prompts[i], candidate_labels, hypothesis_template=hypothesis_template)
                print(f"Prompt: {prompts[i]}")
                print(f"Top {top_k} features for token '{tokens[i][token_idx+1]}': {feat_idx}")
                print(f"Zero-shot classification result: {result}")

# Evaluate feature specificity
evaluate_feature_specificity(feature_activations, tokens, prompts, top_k=10)
