import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import os


class HintonAutoencoder(nn.Module):
    """
    Reference:
    Hinton, G. E., & Salakhutdinov, R. R. (2006). 
    Reducing the dimensionality of data with neural networks. science, 313(5786), 504-507.
    
    Architecture:
    - Encoder: 784 -> 1000 -> 500 -> 250 -> 30
    - Decoder: 30 -> 250 -> 500 -> 1000 -> 784
    """
    def __init__(self):
        super(HintonAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 30)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 784),
            nn.Sigmoid()  # Output between 0 and 1 for MNIST pixels
        )
        
    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(model, device, train_loader, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            _, decoded = model(data)
            loss = criterion(decoded, data.view(-1, 784))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return train_losses

def visualize_reconstruction(model, device, test_loader, save_path=None):
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))[0][:10].to(device)
        _, decoded = model(data)
        
        fig, axes = plt.subplots(2, 10, figsize=(15, 4))

        fig.text(0.05, 0.75, 'Original', fontsize=12, fontweight='bold')
        fig.text(0.05, 0.3, 'Reconstructed', fontsize=12, fontweight='bold')
        
        for i in range(10):
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(decoded[i].cpu().view(28, 28), cmap='gray')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

def plot_losses(losses, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def get_device(device_arg):
    if device_arg == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_arg == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    return device

def main():
    parser = argparse.ArgumentParser(description='Train Hinton\'s Autoencoder on MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size (default: 128)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'], default='mps',
                      help='device to use for training (default: mps)')
    parser.add_argument('--save-model', action='store_true', default=False, help='save the trained model')
    parser.add_argument('--output-dir', type=str, default='results', help='directory to save results')
    args = parser.parse_args()

    device = get_device(args.device)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load and Structure MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the model
    model = HintonAutoencoder().to(device)
    print(model)
    print("Starting training...")
    print(f"Using device: {device}")
    losses = train_autoencoder(model, device, train_loader, num_epochs=args.epochs)

    # Save trained model
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'autoencoder.pth'))
        print(f"Model saved to {os.path.join(args.output_dir, 'autoencoder.pth')}")

    # Generate and save visualizations
    print("Generating visualizations...")
    visualize_reconstruction(model, device, test_loader, 
                           save_path=os.path.join(args.output_dir, 'reconstructions.png'))
    plot_losses(losses, save_path=os.path.join(args.output_dir, 'training_loss.png'))
    
    print(f"Results saved in {args.output_dir}")

if __name__ == '__main__':
    # python hae.py --epochs 100 --batch-size 256 --save-model --output-dir ./hae_results
    main()