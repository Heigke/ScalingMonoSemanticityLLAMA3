from tqdm import tqdm
import re
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os

# Function to extract text between "### Human:" and the first "### Assistant:"
def extract_human_assistant_text(text):
    pattern = r'### Human:(.*?)### Assistant:'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Define the Sparse Autoencoder class
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
        # Reconstruction loss (L2 loss)
        recon_loss = nn.MSELoss()(decoded, x)

        # Sparsity penalty (L1 loss with L2 norm of decoder weights)
        l1_penalty = torch.sum(torch.abs(encoded) * torch.norm(decoder_weights, dim=0))

        # Total loss
        total_loss = recon_loss + self.lambda_reg * l1_penalty
        return total_loss

def train_autoencoder(data, input_dim, hidden_dim, lambda_reg, learning_rate, num_epochs, batch_size, checkpoint_dir, checkpoint_interval=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(input_dim, hidden_dim, lambda_reg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load latest checkpoint if available
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from {latest_checkpoint}, starting at epoch {start_epoch}")
    else:
        start_epoch = 0

    # Data preprocessing: normalize the dataset
    #data = (data - data.mean()) / data.std()
    #data = data.to(device)
    #data = (data - data.mean(dim=1, keepdim=True)) / data.std(dim=1, keepdim=True)
    # Calculate mean and std across batches and tokens for each feature
    means = data.mean(dim=(0, 1), keepdim=True)  # Shape will be (1, 1, features)
    stds = data.std(dim=(0, 1), keepdim=True)    # Shape will be (1, 1, features)
    #torch.save({'means': means, 'stds': stds}, "./means_std_file.pth")
    # Normalize the data
    data = (data - means) / stds
    #torch.save({"data":data},"60000To80000_data.pth")
    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        for batch in tqdm(dataloader):
            batch = batch[0].to(device)
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            loss = model.loss_function(batch, decoded, encoded, model.decoder.weight)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    return model

# Function
# Function to extract activations from the middle layer of the transformer model
def extract_activations(model, tokenizer, texts, middle_layer, max_length=512, batch_size=8):
    all_activations = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states[middle_layer].detach().cpu().to(torch.float32)  # Move to CPU and set dtype for autoencoder training
        #activations = (activations - activations.mean(dim=2, keepdim=True)) / activations.std(dim=2, keepdim=True)
        all_activations.append(activations)
    return torch.cat(all_activations, dim=0)

# Load model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/mnt/hdd1")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="/mnt/hdd1"
)
tokenizer.pad_token = tokenizer.eos_token

# Example usage
if __name__ == "__main__":
    # Load dataset
    ds = load_dataset("DanFosing/wizardlm-vicuna-guanaco-uncensored")
    
    print("Starting preprocessing")
    extracted_texts = []
    j=0
    for entry in tqdm(ds['train']):
        j=j+1
        if j == 90000:
         break
        elif j < 70000:
         continue
        text = entry['text']  # Adjust if the column name is different
        extracted_text = extract_human_assistant_text(text)
        if extracted_text:
            chat = [{"role": "user", "content": extracted_text}]
            form_text = tokenizer.apply_chat_template(chat, tokenize=False)
            extracted_texts.append(form_text)
    print("preprocess done")
    # Extract activations from the middle layer of the transformer model
    num_layers = model.config.num_hidden_layers
    middle_layer = num_layers // 2  # Specify the middle layer
    print("extract act")
    activations = extract_activations(model, tokenizer, extracted_texts, middle_layer, batch_size=8)
    print("done extract act")
    # Train the autoencoder on the extracted activations
    input_dim = activations.shape[-1]  # The dimension of the activations
    hidden_dim = input_dim * 8  # Example hidden dimension (8x expansion of input_dim)
    lambda_reg = 5  # Example regularization coefficient
    learning_rate = 5e-5
    num_epochs = 4000
    batch_size = 100
    print("creating save dir")
    # Create a directory to save checkpoints
    checkpoint_dir = "/mnt/hdd1/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("starting train ae")
    # Train the autoencoder
    trained_model = train_autoencoder(activations, input_dim, hidden_dim, lambda_reg, learning_rate, num_epochs, batch_size, checkpoint_dir)
    print("done train ae")
    # Example forward pass
    example_input = activations[0:1].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    encoded, decoded = trained_model(example_input)
    print("Encoded: ", encoded)
    print("Decoded: ", decoded) 