import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, train_loader, val_loader, device,
                 lr=1e-3, epochs=10, patience=3,
                 checkpoint_path='best_model.pt',
                 plot_path='training_plot.png',
                 optimizer=None, scheduler=None,
                 resume=False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = scheduler
            
        self.epochs = epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.plot_path = plot_path

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.no_improvement_count = 0

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

        if resume:
            self.load_checkpoint()
        else:
            print(f"Starting fresh training with model {self.model.__class__.__name__}")

    def train(self):
        total_epochs = self.current_epoch + self.epochs
        print(f"Starting training at epoch {self.current_epoch + 1} out of {total_epochs}")

        for epoch in range(self.current_epoch + 1, total_epochs + 1):
            print(f"\nEpoch {epoch}/{total_epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loss <= self.best_val_loss:
                print(f"Validation loss improved: {self.best_val_loss:.4f} --> {val_loss:.4f}. Saving checkpoint.")
                self.best_val_loss = val_loss
                self.no_improvement_count = 0
                self.current_epoch = epoch
                self.save_checkpoint()
            else:
                self.no_improvement_count += 1
                print(f"No improvement for {self.no_improvement_count} epochs...")

            if self.no_improvement_count >= self.patience:
                print("\nEarly stopping triggered. Loading best checkpoint...")
                self.load_checkpoint()
                break

        print("\nTraining finished. Loading best model...")
        self.load_checkpoint()
        self.plot_training()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for (padded_word_tensors, padded_vowel_masks, lengths, norm_char_pos, norm_vowel_pos), stress_labels in tqdm(self.train_loader, desc='Train'):

            padded_word_tensors = padded_word_tensors.to(self.device)
            padded_vowel_masks = padded_vowel_masks.to(self.device)
            lengths = lengths.to(self.device)
            norm_char_pos = norm_char_pos.to(self.device)
            norm_vowel_pos = norm_vowel_pos.to(self.device)
            stress_labels = stress_labels.to(self.device)

            logits_list = self.model((padded_word_tensors, padded_vowel_masks, lengths, norm_char_pos, norm_vowel_pos))

            losses = []
            correct_preds = 0

            for i in range(len(logits_list)):
                logits = logits_list[i]
                target = stress_labels[i] - 1    # 0-based индекс слога

                loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                losses.append(loss)

                pred = logits.argmax().item()
                if pred == target.item():
                    correct_preds += 1

            if len(losses) == 0:
                continue

            batch_loss = torch.stack(losses).mean()

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item() * len(logits_list)
            total_correct += correct_preds
            total_samples += len(logits_list)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
        return avg_loss, accuracy

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for (padded_word_tensors, padded_vowel_masks, lengths, norm_char_pos, norm_vowel_pos), stress_labels in tqdm(self.val_loader, desc='Validate'):
            padded_word_tensors = padded_word_tensors.to(self.device)
            padded_vowel_masks = padded_vowel_masks.to(self.device)
            lengths = lengths.to(self.device)
            norm_char_pos = norm_char_pos.to(self.device)
            norm_vowel_pos = norm_vowel_pos.to(self.device)
            stress_labels = stress_labels.to(self.device)

            logits_list = self.model((padded_word_tensors, padded_vowel_masks, lengths, norm_char_pos, norm_vowel_pos))

            losses = []
            correct_preds = 0

            for i in range(len(logits_list)):
                logits = logits_list[i]
                target = stress_labels[i] - 1

                if logits.numel() == 0:
                    continue

                loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                losses.append(loss)

                pred = logits.argmax().item()
                if pred == target.item():
                    correct_preds += 1

            if len(losses) == 0:
                continue

            batch_loss = torch.stack(losses).mean()

            total_loss += batch_loss.item() * len(logits_list)
            total_correct += correct_preds
            total_samples += len(logits_list)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
        return avg_loss, accuracy

    def save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'no_improvement_count': self.no_improvement_count,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Saved new best model to {self.checkpoint_path}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                self.no_improvement_count = checkpoint.get('no_improvement_count', 0)
                self.train_losses = checkpoint.get('train_losses', [])
                self.val_losses = checkpoint.get('val_losses', [])
                self.val_accuracies = checkpoint.get('val_accuracies', [])

                print(f"Loaded checkpoint from {self.checkpoint_path}. Resuming from epoch {self.current_epoch + 1}.")
            except RuntimeError as e:
                print(f"Error loading checkpoint: {e}")
                print("Checkpoint not compatible with current model. Starting from scratch.")
                self.current_epoch = 0
        else:
            print(f"No checkpoint found at {self.checkpoint_path}")
            self.current_epoch = 0

    def plot_training(self):
        if not self.train_losses:
            print("No training data to plot.")
            return
        
        epochs_range = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.train_losses, label='Train Loss')
        plt.plot(epochs_range, self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss per Epoch')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.val_accuracies, label='Val Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Validation Accuracy per Epoch')

        plt.tight_layout()
        plt.savefig(self.plot_path)
        print(f"Training plots saved to {self.plot_path}")
        plt.show()
