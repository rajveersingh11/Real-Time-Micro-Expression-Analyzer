import os
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, save_dir="models"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        
        self.best_val_f1 = 0.0
        os.makedirs(self.save_dir, exist_ok=True)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            
            train_loss = self.train_one_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
                
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")
            
            # Save last model
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_model.pt"))
            
            # Save best model based on F1 Score
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
                print(f"New best model saved with F1: {val_f1:.4f}")

