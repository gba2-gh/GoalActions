import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, CLIPTextModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from decord import VideoReader, cpu
from PIL import Image
import gc
import json

class CLIPSportsNetwork(torch.nn.Module):
    """
    Enhanced version of your existing CLIPNetwork with custom heads for goal detection
    and multi-label action classification
    """
    def __init__(self, num_action_classes=10, dropout_rate=0.3):
        super().__init__()
        
        # Vision tower (same as your existing code)
        vision_tower_name = "openai/clip-vit-large-patch14"
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, low_cpu_mem_usage=True)
        
        # Text encoder for captions (new addition)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        
        feat_dim = 1024
        text_dim = 768  # CLIP text encoder output dimension
        
        # Intermediate layer (same as your existing code)
        self.inter = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )
        
        # NEW: Custom heads for your goal detection task
        # Multimodal fusion layer - adjusted for correct dimensions
        self.multimodal_fusion = nn.Sequential(
            nn.LayerNorm(feat_dim + text_dim),  # Vision (1024) + Text (768) features
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim + text_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Goal/No-goal classification head
        self.goal_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)  # Binary: goal vs no-goal
        )
        
        # Multi-label tactical action head
        self.tactical_action_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_action_classes)  # Multi-label classification
        )


        # self.goal_head = nn.Sequential(
        # nn.LayerNorm(feat_dim),      # Input: 1024 (pure vision)
        # nn.Dropout(dropout_rate),
        # nn.Linear(1024, 512),        # LARGER intermediate layer
        # nn.ReLU(),
        # nn.Dropout(dropout_rate),
        # nn.Linear(512, 256),         # Additional layer for complexity
        # nn.ReLU(),
        # nn.Dropout(dropout_rate),
        # nn.Linear(256, 2)            # Binary classification
        # )

        # self.tactical_action_head = nn.Sequential(
        # nn.LayerNorm(feat_dim),      # Input: 1024 (pure vision)
        # nn.Dropout(dropout_rate),
        # nn.Linear(1024, 512),        # LARGER intermediate layer
        # nn.ReLU(),
        # nn.Dropout(dropout_rate),
        # nn.Linear(512, 256),         # Additional layer for complexity
        # nn.ReLU(),
        # nn.Dropout(dropout_rate),
        # nn.Linear(256, num_action_classes)  # Multi-label classification
        # )

        
        # Initialize new heads
        self._init_custom_heads()
    
    def _init_custom_heads(self):
        """Initialize custom head weights"""
        for module in [self.multimodal_fusion, self.goal_head, self.tactical_action_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text captions"""
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return text_outputs.pooler_output
    
    def forward(self, video, input_ids=None, attention_mask=None, return_multimodal=False):
        """
        Forward pass with optional text input for multimodal features
        
        Args:
            video: Video frames tensor [batch_size, num_frames, 3, 224, 224] or [num_frames, 3, 224, 224]
            input_ids: Text token ids (optional)
            attention_mask: Text attention mask (optional)
            return_multimodal: Whether to return multimodal predictions
        
        Returns:
            If return_multimodal=False: (out_off, out_act, video_features) - your existing outputs
            If return_multimodal=True: (out_off, out_act, video_features, goal_logits, tactical_logits)
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
        
        # Vision processing (same as your existing code)
        out = self.vision_tower(video, output_hidden_states=True)
        
        # Extract features for spatio-temporal processing
        select_hidden_state_layer = -2
        select_hidden_state = out.hidden_states[select_hidden_state_layer]
        batch_features = select_hidden_state[:, 1:]
        video_features = batch_features.detach().cpu()
        
        # Reshape pooler output back to [batch_size * num_frames, hidden_size]
        pooler_output = out.pooler_output  # [batch_size * num_frames, hidden_size]
        
        # Reshape to [batch_size, num_frames, hidden_size] and then average over frames
        pooler_output = pooler_output.view(batch_size, num_frames, -1)
        vision_feat = torch.mean(pooler_output, dim=1)  # [batch_size, hidden_size]
        
        vision_feat = self.inter(vision_feat)
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        
       # if not return_multimodal or input_ids is None:
            #return out_off.squeeze(), out_act.squeeze(), video_features
        
        # NEW: Multimodal processing for goal detection
        text_feat = self.encode_text(input_ids, attention_mask)
        
        # Ensure text features have the same batch dimension as vision features
        if text_feat.shape[0] != vision_feat.shape[0]:
            text_feat = text_feat.mean(dim=0, keepdim=True)
        
        # Fuse vision and text features
        #multimodal_feat = torch.cat([vision_feat, text_feat], dim=1)
        #fused_feat = self.multimodal_fusion(multimodal_feat)
        
        # New predictions
        goal_logits = self.goal_head(vision_feat)
        tactical_logits = self.tactical_action_head(vision_feat)
        
        
        
        return None, None, video_features, goal_logits, tactical_logits


class SportsVideoDataset(Dataset):
    def __init__(self, video_paths, captions, outcomes, action_vectors, 
                 image_processor, tokenizer, max_frames=25):
        """
        Dataset compatible with your existing video processing approach
        
        Args:
            video_paths: List of paths to video files
            captions: List of text captions for each video
            outcomes: List of binary labels (0: no-goal, 1: goal)
            action_vectors: List of multi-label vectors for tactical actions
            image_processor: CLIP image processor
            tokenizer: CLIP tokenizer
            max_frames: Maximum number of frames to extract
        """
        self.video_paths = video_paths
        self.captions = captions
        self.outcomes = outcomes
        self.action_vectors = action_vectors
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_frames(self, video_path):
        """
        Load video frames using decord (similar to your existing approach)
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        
        # Focus on key action timeframe (similar to your existing logic)
        # For 5-second clips, focus on middle 3 seconds where action likely occurs
        start_frame = max(0, int(total_frame_num * 0.2))
        end_frame = min(total_frame_num, int(total_frame_num * 0.8))

        start_frame = 0
        end_frame = total_frame_num
        
        #print(f"Video {video_path}: {total_frame_num} total frames, using frames {start_frame}-{end_frame}")
        
        
        # Sample frames within this range
        if end_frame - start_frame <= self.max_frames:
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Sample evenly within the key timeframe
            frame_indices = np.linspace(start_frame, end_frame-1, self.max_frames, dtype=int)
            #print(f"  Sampling {self.max_frames} frames evenly across full duration")

        if len(vr) > 1:
            fps_estimate = len(vr) / 5.0  # Assuming 5-second clips
            actual_coverage = len(frame_indices) / fps_estimate
            #print(f"  Estimated FPS: {fps_estimate:.1f}, Temporal coverage: {actual_coverage:.2f} seconds")
         
        
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
        
        # Process frames (similar to your existing preprocessing)
        pixel_values = self.image_processor(frames, return_tensors='pt')['pixel_values']
        
        # Process text caption with fixed padding
        text_inputs = self.tokenizer(
            self.captions[idx],
            return_tensors="pt",
            padding='max_length',  # This ensures consistent length
            truncation=True,
            max_length=77
        )
        
        return {
            'pixel_values': pixel_values.squeeze(1),  # Remove extra batch dimension
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'outcome': torch.tensor(self.outcomes[idx], dtype=torch.long),
            'action_vector': torch.tensor(self.action_vectors[idx], dtype=torch.float32)
        }

class SportsAnalysisTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.goal_criterion = nn.CrossEntropyLoss()
        self.tactical_criterion = nn.BCEWithLogitsLoss()
        
        # Separate optimizers for different parts
        # Your existing heads
        existing_params = list(self.model.vision_tower.parameters()) + \
                         list(self.model.inter.parameters()) 
                         #list(self.model.fc_offence.parameters()) + \
                         #list(self.model.fc_action.parameters())
        
        # New multimodal heads
        new_params = list(self.model.text_encoder.parameters()) + \
                    list(self.model.multimodal_fusion.parameters()) + \
                    list(self.model.goal_head.parameters()) + \
                    list(self.model.tactical_action_head.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': existing_params, 'lr': 1e-5},  # Lower LR for pre-trained parts
            {'params': new_params, 'lr': 1e-5}       # Higher LR for new components
        ], weight_decay=0.01)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
    
    def train_epoch(self, dataloader, goal_weight=1.0, tactical_weight=1.0):
        self.model.train()
        total_loss = 0
        goal_preds, goal_targets = [], []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            outcomes = batch['outcome'].to(self.device)
            action_vectors = batch['action_vector'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with multimodal features
            out_off, out_act, video_features, goal_logits, tactical_logits = self.model(
                pixel_values, input_ids, attention_mask, return_multimodal=True
            )
            
            # Calculate losses for new heads
            goal_loss = self.goal_criterion(goal_logits, outcomes)
            tactical_loss = self.tactical_criterion(tactical_logits, action_vectors)
            
            # Combined loss
            total_batch_loss = goal_weight * goal_loss + tactical_weight * tactical_loss
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # Store predictions (handle batch dimension)
            if goal_logits.dim() == 1:  # Single sample
                goal_preds.extend([torch.argmax(goal_logits).item()])
                goal_targets.extend([outcomes.item()])
            else:  # Batch
                goal_preds.extend(torch.argmax(goal_logits, dim=1).cpu().numpy())
                goal_targets.extend(outcomes.cpu().numpy())
            
            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        goal_acc = accuracy_score(goal_targets, goal_preds)
        
        return avg_loss, goal_acc
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        goal_preds, goal_targets = [], []
        tactical_preds, tactical_targets = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outcomes = batch['outcome'].to(self.device)
                action_vectors = batch['action_vector'].to(self.device)
                
                # Forward pass
                out_off, out_act, video_features, goal_logits, tactical_logits = self.model(
                    pixel_values, input_ids, attention_mask, return_multimodal=True
                )
                
                # Calculate losses
                goal_loss = self.goal_criterion(goal_logits, outcomes)
                tactical_loss = self.tactical_criterion(tactical_logits, action_vectors)
                total_loss += (goal_loss + tactical_loss).item()
                
                
                # Store predictions
                goal_preds.extend(torch.argmax(goal_logits, dim=1).cpu().numpy())
                goal_targets.extend(outcomes.cpu().numpy())
                
                tactical_pred = torch.sigmoid(tactical_logits) > 0.5
                tactical_preds.extend(tactical_pred.cpu().numpy())
                tactical_targets.extend(action_vectors.cpu().numpy())
                
                # Memory cleanup
                gc.collect()
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        goal_acc = accuracy_score(goal_targets, goal_preds)
        goal_f1 = f1_score(goal_targets, goal_preds, average='weighted')
        
        # Multi-label metrics
        tactical_preds = np.array(tactical_preds)
        tactical_targets = np.array(tactical_targets)
        tactical_f1 = f1_score(tactical_targets, tactical_preds, average='micro')
        tactical_f1 = 0
        return {
            'loss': avg_loss,
            'goal_accuracy': goal_acc,
            'goal_f1': goal_f1,
            'tactical_f1': tactical_f1
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
                  f"Val Tactical F1: {val_metrics['tactical_f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'epoch': epoch
                }, 'best_enhanced_sports_clip_model.pth')
                print("New best model saved!")
            
            print("-" * 50)

# Custom collate function to handle variable frame counts
def custom_collate_fn(batch):
    """Custom collate function to handle batching of video data"""
    
    # Find the maximum number of frames in this batch
    max_frames = max([item['pixel_values'].shape[0] for item in batch])
    
    batched_data = {
        'pixel_values': [],
        'input_ids': [],
        'attention_mask': [],
        'outcome': [],
        'action_vector': []
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
        batched_data['input_ids'].append(item['input_ids'])
        batched_data['attention_mask'].append(item['attention_mask'])
        batched_data['outcome'].append(item['outcome'])
        batched_data['action_vector'].append(item['action_vector'])
    
    # Stack all tensors
    return {
        'pixel_values': torch.stack(batched_data['pixel_values']),
        'input_ids': torch.stack(batched_data['input_ids']),
        'attention_mask': torch.stack(batched_data['attention_mask']),
        'outcome': torch.stack(batched_data['outcome']),
        'action_vector': torch.stack(batched_data['action_vector'])
    }

def create_dataset(path):
    # Initialize processors
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
    tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14')

    with open(path, mode='r', encoding = 'utf-8') as f:
        all_plays = json.load(f)

    video_paths = []
    video_root = 'results/video_goal_output_60_updated'
    captions = []
    outcomes = []
    action_vectors = []
    for play in all_plays:
        video_paths.append(f"{video_root}/action_{play['id']}/clip.mp4")
        captions.append(play['caption'])
        outcome = 1 if play['label']  else 0
        outcomes.append(outcome)
        action_vectors.append(play['actions_vector'])
        
    # Create dataset
    dataset = SportsVideoDataset(
        video_paths, captions, outcomes, action_vectors, 
        image_processor, tokenizer
    )

    return dataset


# Example usage
def main():

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
    
    # Initialize enhanced model
    model = CLIPSportsNetwork(num_action_classes=10)
    
    # Load your existing weights if available
    #checkpoint = torch.load('best_enhanced_sports_clip_model_3epochs.pth')
    #model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    trainer = SportsAnalysisTrainer(model)
    
    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=10)

if __name__ == "__main__":
    main()