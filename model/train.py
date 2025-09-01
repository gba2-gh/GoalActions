import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import json

class SportsVideoDataset(Dataset):
    def __init__(self, video_paths, captions, outcomes, action_vectors, processor, max_frames=8):
        """
        Args:
            video_paths: List of paths to video files
            captions: List of text captions for each video
            outcomes: List of binary labels (0: no-goal, 1: goal)
            action_vectors: List of multi-label vectors for tactical actions
            processor: CLIP processor for preprocessing
            max_frames: Maximum number of frames to extract per video
        """
        self.video_paths = video_paths
        self.captions = captions
        self.outcomes = outcomes
        self.action_vectors = action_vectors
        self.processor = processor
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if total_frames <= self.max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        # Extract frames from video
        frames = self.extract_frames(self.video_paths[idx])
        
                # For multiple frames, we'll average them or take the middle frame
        # Here we'll take the middle frame for simplicity
        if len(frames) > 0:
            middle_idx = len(frames) // 2
            representative_frame = frames[middle_idx]
        else:
            # If no frames extracted, create a blank frame
            representative_frame = np.zeros((224, 224, 3), dtype=np.uint8)


        # Process frames and text
        inputs = self.processor(
            text=self.captions[idx],
            images=representative_frame,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Remove batch dimension added by processor
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'outcome': torch.tensor(self.outcomes[idx], dtype=torch.long),
            'action_vector': torch.tensor(self.action_vectors[idx], dtype=torch.float32)
        }

class CLIPSportsAnalyzer(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", 
                 num_action_classes=20, dropout_rate=0.3):
        """
        Args:
            clip_model_name: Pre-trained CLIP model to use
            num_action_classes: Number of action classes for multi-label classification
            dropout_rate: Dropout rate for custom heads
        """
        super(CLIPSportsAnalyzer, self).__init__()
        
        # Load pre-trained CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Get the dimension of CLIP's joint embedding space
        self.embed_dim = self.clip.projection_dim
        
        # Custom heads
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # Binary classification: goal vs no-goal
        )
        
        self.action_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_action_classes)  # Multi-label classification
        )
        
        # Initialize custom head weights
        self._init_custom_heads()
    
    def _init_custom_heads(self):
        """Initialize custom head weights"""
        for module in [self.outcome_head, self.action_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, input_ids, attention_mask, pixel_values, return_embeddings=False):
        # Get CLIP outputs
        clip_outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # Use multimodal embeddings (image-text joint representation)
        # Average pool the image and text features
        image_embeds = clip_outputs.image_embeds
        text_embeds = clip_outputs.text_embeds
        
        # Combine image and text embeddings
        joint_embeds = (image_embeds + text_embeds) / 2
        
        # Pass through custom heads
        outcome_logits = self.outcome_head(joint_embeds)
        action_logits = self.action_head(joint_embeds)
        
        if return_embeddings:
            return outcome_logits, action_logits, joint_embeds
        
        return outcome_logits, action_logits

class SportsAnalysisTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.outcome_criterion = nn.CrossEntropyLoss()
        self.action_criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer - different learning rates for CLIP backbone vs custom heads
        clip_params = []
        custom_params = []
        
        for name, param in self.model.named_parameters():
            if 'clip' in name:
                clip_params.append(param)
            else:
                custom_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': clip_params, 'lr': 1e-5},    # Lower LR for pre-trained CLIP
            {'params': custom_params, 'lr': 1e-3}   # Higher LR for custom heads
        ], weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
    
    def train_epoch(self, dataloader, outcome_weight=1.0, action_weight=1.0):
        self.model.train()
        total_loss = 0
        outcome_preds, outcome_targets = [], []
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            outcomes = batch['outcome'].to(self.device)
            action_vectors = batch['action_vector'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outcome_logits, action_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            # Calculate losses
            outcome_loss = self.outcome_criterion(outcome_logits, outcomes)
            action_loss = self.action_criterion(action_logits, action_vectors)
            
            # Combined loss
            total_batch_loss = (outcome_weight * outcome_loss + 
                              action_weight * action_loss)
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # Store predictions for metrics
            outcome_preds.extend(torch.argmax(outcome_logits, dim=1).cpu().numpy())
            outcome_targets.extend(outcomes.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        outcome_acc = accuracy_score(outcome_targets, outcome_preds)
        
        return avg_loss, outcome_acc
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        outcome_preds, outcome_targets = [], []
        action_preds, action_targets = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                outcomes = batch['outcome'].to(self.device)
                action_vectors = batch['action_vector'].to(self.device)
                
                # Forward pass
                outcome_logits, action_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
                
                # Calculate losses
                outcome_loss = self.outcome_criterion(outcome_logits, outcomes)
                action_loss = self.action_criterion(action_logits, action_vectors)
                total_loss += (outcome_loss + action_loss).item()
                
                # Store predictions
                outcome_preds.extend(torch.argmax(outcome_logits, dim=1).cpu().numpy())
                outcome_targets.extend(outcomes.cpu().numpy())
                
                # For multi-label, use sigmoid threshold
                action_pred = torch.sigmoid(action_logits) > 0.5
                action_preds.extend(action_pred.cpu().numpy())
                action_targets.extend(action_vectors.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        outcome_acc = accuracy_score(outcome_targets, outcome_preds)
        outcome_f1 = f1_score(outcome_targets, outcome_preds, average='weighted')
        
        # Multi-label metrics
        action_preds = np.array(action_preds)
        action_targets = np.array(action_targets)
        action_f1 = f1_score(action_targets, action_preds, average='micro')
        
        return {
            'loss': avg_loss,
            'outcome_accuracy': outcome_acc,
            'outcome_f1': outcome_f1,
            'action_f1': action_f1
        }
    
    def train(self, train_loader, val_loader, num_epochs=10, 
              outcome_weight=1.0, action_weight=1.0):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(
                train_loader, outcome_weight, action_weight
            )
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Outcome Acc: {val_metrics['outcome_accuracy']:.4f}, "
                  f"Val Action F1: {val_metrics['action_f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'epoch': epoch
                }, 'best_sports_clip_model.pth')
                print("New best model saved!")
            
            print("-" * 50)

# Example usage
def main():
    # Initialize processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Example data (replace with your actual data loading)
    video_paths = ["video1.mp4", "video2.mp4"]  # Your video file paths
    captions = ["Player shoots towards goal", "Defensive play in midfield"]
    outcomes = [1, 0]  # 1 for goal, 0 for no-goal
    action_vectors = [
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Multi-hot encoding
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # Create dataset and dataloader
    dataset = SportsVideoDataset(
        video_paths, captions, outcomes, action_vectors, processor
    )
    
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model and trainer
    model = CLIPSportsAnalyzer(num_action_classes=20)
    trainer = SportsAnalysisTrainer(model)
    print('finished')
    # Train model
    trainer.train(train_loader, train_loader, num_epochs=10)

if __name__ == "__main__":
    main()