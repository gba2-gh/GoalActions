import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from decord import VideoReader, cpu
from PIL import Image
import gc
import json
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class CLIPSportsNetwork(torch.nn.Module):
    """
    Vision-only CLIP network for goal detection
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Vision tower only
        vision_tower_name = "openai/clip-vit-large-patch14"
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, low_cpu_mem_usage=True)
        
        feat_dim = 1024
        
        # Intermediate layer (same as your existing code)
        self.inter = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, feat_dim),
            nn.ReLU(), 
            nn.Linear(feat_dim, feat_dim),
        )
        
        # Goal/No-goal classification head only
        self.goal_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # Binary: goal vs no-goal
        )
        
        # Initialize new head
        self._init_custom_heads()
    
    def _init_custom_heads(self):
        """Initialize custom head weights"""
        for layer in self.goal_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, video):
        """
        Forward pass with vision features only
        
        Args:
            video: Video frames tensor [batch_size, num_frames, 3, 224, 224] or [num_frames, 3, 224, 224]
        
        Returns:
            goal_logits: Goal detection predictions
        """
        # Handle different input shapes
        if video.dim() == 5:  # [batch_size, num_frames, 3, 224, 224]
            batch_size, num_frames = video.shape[0], video.shape[1]
            # Reshape to [batch_size * num_frames, 3, 224, 224]
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4])
        elif video.dim() == 4:  # [num_frames, 3, 224, 224]
            batch_size, num_frames = 1, video.shape[0]
        else:
            raise ValueError(f"Unexpected video tensor shape: {video.shape}")
        
        # Vision processing
        out = self.vision_tower(video, output_hidden_states=True)
        
        # Extract features for spatio-temporal processing
        # select_hidden_state_layer = -2
        # select_hidden_state = out.hidden_states[select_hidden_state_layer]
        # batch_features = select_hidden_state[:, 1:]
        # video_features = batch_features.detach().cpu()
        
        # Reshape pooler output back to [batch_size * num_frames, hidden_size]
        pooler_output = out.pooler_output  # [batch_size * num_frames, hidden_size]
        
        # Reshape to [batch_size, num_frames, hidden_size] and then average over frames
        pooler_output = pooler_output.view(batch_size, num_frames, -1)
        vision_feat = torch.mean(pooler_output, dim=1)  # [batch_size, hidden_size]
        
        vision_feat = self.inter(vision_feat)
        
        # Clean up memory
        #gc.collect()
        #torch.cuda.empty_cache()
        
        # Goal prediction from vision features only
        goal_logits = self.goal_head(vision_feat)
        
        return goal_logits


class SportsVideoDataset(Dataset):
    def __init__(self, video_paths, outcomes, image_processor, max_frames=25):
        """
        Dataset for goal detection using vision only
        
        Args:
            video_paths: List of paths to video files
            outcomes: List of binary labels (0: no-goal, 1: goal)
            image_processor: CLIP image processor
            max_frames: Maximum number of frames to extract
        """
        self.video_paths = video_paths
        self.outcomes = outcomes
        self.image_processor = image_processor
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def biased_frame_sampling(self, num_frames: int, start_frame: int = 0,target_frames: int = 20) -> np.ndarray:
        if num_frames <= target_frames:
            return np.arange(num_frames)  # just take all frames
        
        # Create a nonlinear "time curve" that spends more samples at the end
        # Exponential bias: small numbers early, dense at the end
        curve = np.linspace(0, 1, target_frames) ** 2  
        
        total_frames = num_frames - start_frame
        # Scale to the frame range
        indices = (curve * (total_frames - 1)).astype(int) + start_frame
        
        return np.unique(indices)  # ensure sorted & unique

    
    def load_video_frames(self, video_path):
        """
        Load video frames using decord
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        
        # Focus on key action timeframe
        #start_frame = max(0, int(total_frame_num * 0.5))
        #end_frame = min(total_frame_num, int(total_frame_num * 1))
        #start_frame = 0
        #end_frame = total_frame_num
        
        # # Sample frames within this range
        # if end_frame - start_frame <= self.max_frames:
        #     frame_indices = list(range(start_frame, end_frame))
        # else:
        #     # Sample evenly within the key timeframe
        #     frame_indices = np.linspace(start_frame, end_frame-1, self.max_frames, dtype=int)

        frame_indices = self.biased_frame_sampling(total_frame_num,50, 20)
        
        img_array = vr.get_batch(frame_indices).asnumpy()
        
        # Convert to PIL Images and resize
        frames = []
        for i in range(img_array.shape[0]):
            frame = Image.fromarray(img_array[i])
            frames.append(frame)
        
        return frames
    
    def __getitem__(self, idx):
        # Load video frames
        frames = self.load_video_frames(self.video_paths[idx])
        
        # Process frames
        pixel_values = self.image_processor(frames, return_tensors='pt')['pixel_values']
        
        return {
            'pixel_values': pixel_values.squeeze(1),  # Remove extra batch dimension
            'outcome': torch.tensor(self.outcomes[idx], dtype=torch.long)
        }

class SportsAnalysisTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', gradient_accumulation_steps=8):
        self.model = model.to(device)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps  # Simulate batch_size = 2 * 8 = 16
        
        
        # Loss function for goal detection only
        self.goal_criterion = nn.CrossEntropyLoss()
        
        # Optimizer for vision-only model
        existing_params = list(self.model.vision_tower.parameters()) + \
                         list(self.model.inter.parameters()) 
        
        new_params = list(self.model.goal_head.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': existing_params, 'lr': 1e-5},  # Lower LR for pre-trained parts
            {'params': new_params, 'lr': 1e-5}       # Higher LR for new components
        ], weight_decay=0.01)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        goal_preds, goal_targets = [], []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            outcomes = batch['outcome'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with vision only
            goal_logits = self.model(pixel_values)
            
            # Calculate loss for goal detection only
            goal_loss = self.goal_criterion(goal_logits, outcomes)
            
             # Scale loss by accumulation steps (important!)
            scaled_loss = goal_loss / self.gradient_accumulation_steps
            # Backward pass (gradients accumulate)
            scaled_loss.backward()

            # Backward pass
            #goal_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            #self.optimizer.step()
            
            total_loss += goal_loss.item()
            
            # Store predictions (handle batch dimension)
            if goal_logits.dim() == 1:  # Single sample
                goal_preds.extend([torch.argmax(goal_logits).item()])
                goal_targets.extend([outcomes.item()])
            else:  # Batch
                goal_preds.extend(torch.argmax(goal_logits, dim=1).cpu().numpy())
                goal_targets.extend(outcomes.cpu().numpy())
            

            # Update weights every gradient_accumulation_steps batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if batch_idx % (10 * self.gradient_accumulation_steps) == 0:
                    print(f"Step {batch_idx}, Loss: {goal_loss.item():.4f}")


            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {goal_loss.item():.4f}")

        # Handle remaining gradients if total batches not divisible by accumulation steps
        if len(dataloader) % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        goal_acc = accuracy_score(goal_targets, goal_preds)
        
        return avg_loss, goal_acc
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        goal_preds, goal_targets = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                outcomes = batch['outcome'].to(self.device)
                
                # Forward pass
                goal_logits = self.model(pixel_values)
                
                # Calculate loss
                goal_loss = self.goal_criterion(goal_logits, outcomes)
                total_loss += goal_loss.item()
                
                # Store predictions
                goal_preds.extend(torch.argmax(goal_logits, dim=1).cpu().numpy())
                goal_targets.extend(outcomes.cpu().numpy())
                


                #print("\nClassification Report (Goals):")
               # print(classification_report(goal_targets, goal_preds, digits=4))


                # Memory cleanup
                gc.collect()
                torch.cuda.empty_cache()
            
        cm = confusion_matrix(goal_targets, goal_preds)
        print("Confusion Matrix (Goals):")
        print(cm)
        print(classification_report(goal_targets, goal_preds, digits=4))
        
        avg_loss = total_loss / len(dataloader)
        goal_acc = accuracy_score(goal_targets, goal_preds)
        goal_f1 = f1_score(goal_targets, goal_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'goal_accuracy': goal_acc,
            'goal_f1': goal_f1
        }
    
    def train(self, train_loader, val_loader, num_epochs=10):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Goal Acc: {val_metrics['goal_accuracy']:.4f}, "
                  f"Val Goal F1: {val_metrics['goal_f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'epoch': epoch
                }, 'best_goal_detection_clip_model.pth')
                print("New best model saved!")
            
            print("-" * 50)

# Custom collate function to handle variable frame counts
def custom_collate_fn(batch):
    """Custom collate function to handle batching of video data"""
    
    # Find the maximum number of frames in this batch
    max_frames = max([item['pixel_values'].shape[0] for item in batch])
    
    batched_data = {
        'pixel_values': [],
        'outcome': []
    }
    
    for item in batch:
        # Pad or truncate video frames to max_frames
        frames = item['pixel_values']
        if frames.shape[0] < max_frames:
            # Pad with the last frame repeated
            padding = frames[-1:].repeat(max_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)
        elif frames.shape[0] > max_frames:
            # Truncate to max_frames
            frames = frames[:max_frames]
        
        batched_data['pixel_values'].append(frames)
        batched_data['outcome'].append(item['outcome'])
    
    # Stack all tensors
    return {
        'pixel_values': torch.stack(batched_data['pixel_values']),
        'outcome': torch.stack(batched_data['outcome'])
    }

def create_dataset(path):
    # Initialize processors
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')

    with open(path, mode='r', encoding = 'utf-8') as f:
        all_plays = json.load(f)

    video_paths = []
    video_root = 'results/video_goal_output_60_updated'
    outcomes = []
    k =0
    for play in all_plays:
        
        video_paths.append(f"{video_root}/action_{play['id']}/clip.mp4")
        outcome = 1 if play['label']  else 0
        outcomes.append(outcome)
        
    # Create dataset (vision only)
    dataset = SportsVideoDataset(
        video_paths, outcomes, image_processor
    )

    return dataset


# Example usage
def main():
    print("new model")

    train_data = create_dataset('results/divided_data/train.json')
    val_data = create_dataset('results/divided_data/val.json')

 
    # Use custom collate function
    train_loader = DataLoader(
        train_data, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    print(f"Data loaders created:")
    print(f"- Training batches: {len(train_loader)}")
    print(f"- Validation batches: {len(val_loader)}")
    
    # Initialize model for goal detection only
    model = CLIPSportsNetwork()
    
    # Load existing weights if available
    #checkpoint = torch.load('14_model.pth')
    #model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    trainer = SportsAnalysisTrainer(model, gradient_accumulation_steps=16)
    
    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=10)

if __name__ == "__main__":
    main()