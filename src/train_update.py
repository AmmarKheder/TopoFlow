
# Add resume capability
def resume_from_checkpoint(model, optimizer, checkpoint_dir='checkpoints'):
    import glob
    checkpoint_files = glob.glob(f'{checkpoint_dir}/checkpoint_epoch_*.pth')
    if not checkpoint_files:
        print('No checkpoint found. Starting fresh training.')
        return 0, None, float('inf')
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f'Resuming from checkpoint {latest_checkpoint}')
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
    metrics = checkpoint['metrics']
    return start_epoch, metrics, best_val_loss

# Main training function

def main():
    from config_manager import ConfigManager
    from dataloader import CAQRADataset
    from model import PM25Model
    
    print('Starting AQ_Net2 Training')
    
    # Load config
    config = ConfigManager('configs/config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    print('Loading datasets...')
