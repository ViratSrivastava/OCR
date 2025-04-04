import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import wandb
import csv

def log_analytics_to_csv(analytics_csv_path, epoch, loss, accuracy):
    """Log analytics to a CSV file."""
    with open(analytics_csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['epoch', 'loss', 'accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write header if file is new
        if csv_file.tell() == 0:
            writer.writeheader()
        
        writer.writerow({'epoch': epoch, 'loss': loss, 'accuracy': accuracy})

def visualize_analytics(analytics_csv_path, analytics_dir):
    """Visualize analytics using Matplotlib, Seaborn, and Plotly."""
    data = pd.read_csv(analytics_csv_path)
    
    # Matplotlib
    plt.figure(figsize=(10, 6))
    plt.plot(data['epoch'], data['loss'], label='Loss')
    plt.plot(data['epoch'], data['accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.savefig(os.path.join(analytics_dir, 'training_loss_accuracy.png'))
    plt.show()
    
    # Seaborn
    sns.set()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=data['epoch'], y=data['loss'], label='Loss')
    sns.lineplot(x=data['epoch'], y=data['accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.savefig(os.path.join(analytics_dir, 'seaborn_training_loss_accuracy.png'))
    plt.show()
    
    # Plotly
    fig = px.line(x=data['epoch'], y=[data['loss'], data['accuracy']], labels={'x': 'Epochs', 'y': 'Values'})
    fig.update_layout(title='Training Loss and Accuracy', xaxis_title='Epochs', yaxis_title='Values')
    fig.write_image(os.path.join(analytics_dir, 'plotly_training_loss_accuracy.png'))  # Requires kaleido
    fig.show()

def log_to_wandb(epoch, loss, accuracy):
    """Log analytics to WandB."""
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })
