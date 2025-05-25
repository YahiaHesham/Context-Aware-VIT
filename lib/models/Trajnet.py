import torch
import torch.nn as nn
from .feature_extractor import build_feature_extractor
from .SubLayers import MultiHeadAttention, FeedForward

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

class TrajEncoder(nn.Module):
    """Encoder for trajectory and speed data"""
    def __init__(self, args, device):
        super(TrajEncoder, self).__init__()
        self.hidden_size_traj = args.hidden_size_traj
        self.hidden_size_sp = args.hidden_size_sp
        self.dropout = args.dropout
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.device = device
        
        # Feature extractors
        self.feature_extractor_traj = build_feature_extractor(args, ft_type='traj')
        self.feature_extractor_speed = build_feature_extractor(args, ft_type='speed')
        
        # Projection layer to match dimensions
        self.combine_proj = nn.Linear(
            args.hidden_size_traj + args.hidden_size_sp,
            args.hidden_size_traj  # Project to match GRU input size
        )
        
        # Transformer encoder layers
        self.EncoderLayer = EncoderLayer(args)
        self.pos_em_enc_traj = self.pos_embed(self.enc_steps, self.hidden_size_traj)
        self.pos_em_enc_sp = self.pos_embed(self.enc_steps, self.hidden_size_sp)
        self.pos_em_dec_traj = self.pos_embed(self.dec_steps, self.hidden_size_traj)
        
        # GRU for sequential processing
        self.traj_gru = nn.GRU(
            input_size=args.hidden_size_traj,
            hidden_size=args.hidden_size_traj,
            num_layers=2,
            batch_first=True,
            dropout=args.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(args.hidden_size_traj, args.d_model_traj)

    def pos_embed(self, length, hidden_size, n=10000):
        P = torch.zeros((length, hidden_size), device=self.device)
        for k in range(length):
            for i in torch.arange(int(hidden_size/2)):
                denominator = torch.pow(n, 2*i/hidden_size)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P

    def forward(self, traj, speed):
    # Extract features
        traj_input = self.feature_extractor_traj(traj) + self.pos_em_enc_traj  # [B, T, 256]
        speed_input = self.feature_extractor_speed(speed) + self.pos_em_enc_sp  # [B, T, 128]
    
    # Combine features
          # [B, T, 384]
    
    # Project to expected dimension for EncoderLayer
        traj_speed = torch.cat((traj_input, speed_input), axis=-1)  # [B, T, 384]
        
        
        encoded = self.EncoderLayer(traj_speed)  # [B, T, 384]
        # print(f"Shape after encoder: {encoded.shape}")
    # Project to GRU input size
        gru_input = encoded[..., :self.hidden_size_traj]  # Take first 256 dims
        output, hidden = self.traj_gru(gru_input)
    
        output = self.output_projection(output)
        
        return output, hidden


class VisualEncoder(nn.Module):
    """Vision Transformer based encoder for visual context"""
    def __init__(self, args, device):
        super(VisualEncoder, self).__init__()
        self.device = device
        self.batch_size = args.batch_size
        
        # Initialize ViT model
        if TIMM_AVAILABLE:
            self.vit = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
            self.vit.head = nn.Identity()
            self.vit_dim = self.vit.num_features
            from timm.data import resolve_model_data_config, create_transform
            config = resolve_model_data_config(self.vit)
            self.vit_transform = create_transform(**config, is_training=False)
        else:
            from torchvision import models, transforms
            self.vit = models.vit_b_16(pretrained=True)
            self.vit.heads = nn.Identity()
            self.vit_dim = self.vit.hidden_dim
            self.vit_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Embeddings for additional tokens
        self.bbx_embed = nn.Linear(4, self.vit_dim)  # 4 for bbox coords
        self.ego_embed = nn.Linear(1, self.vit_dim)  # 1 for ego speed
        
        # GRU for sequential visual features
        self.visual_gru = nn.GRU(
            input_size=self.vit_dim,
            hidden_size=self.vit_dim,
            num_layers=2,
            batch_first=True,
            dropout=args.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.vit_dim, args.d_model_traj)
    
    def forward(self, frames, traj, speed):
        B = traj.shape[0]
        visual_features = []
        
        for t in range(len(frames[0])):
            # Get frames for current timestep
            current_frames = [seq[t] for seq in frames]
            current_frames = torch.stack([self.vit_transform(img) for img in current_frames]).to(self.device)
            
            # Process through ViT
            vit = self.vit
            x = vit.patch_embed(current_frames)
            cls_token = vit.cls_token.expand(B, -1, -1)
            
            # Add trajectory and speed tokens
            t_idx = min(t, traj.shape[1]-1)
            bbx_token = traj[:, t_idx, :]
            ego_token = speed[:, t_idx, :]
            
            bbx_token_emb = self.bbx_embed(bbx_token).unsqueeze(1)
            ego_token_emb = self.ego_embed(ego_token).unsqueeze(1)
            
            # Combine tokens
            x = torch.cat((cls_token, x, bbx_token_emb, ego_token_emb), dim=1)
            
            # Add position embeddings
            pos_embed = vit.pos_embed
            if x.shape[1] > pos_embed.shape[1]:
                extra = x.shape[1] - pos_embed.shape[1]
                pos_embed = torch.cat([
                    pos_embed,
                    pos_embed[:, -1:, :].repeat(1, extra, 1)
                ], dim=1)
            
            x = x + pos_embed[:, :x.shape[1], :]
            x = vit.pos_drop(x)
            
            # Forward through transformer blocks
            for blk in vit.blocks:
                x = blk(x)
            
            x = vit.norm(x)
            
            # Extract CLS token
            cls_out = x[:, 0]
            visual_features.append(cls_out)
        
        # Process sequence of visual features
        visual_features = torch.stack(visual_features, dim=1)
        output, hidden = self.visual_gru(visual_features)
        
        # Project to common dimension
        output = self.output_projection(output)
        
        return output, hidden


class TrajDecoder(nn.Module):
    def __init__(self, args):
        super(TrajDecoder, self).__init__()
        self.d_model = args.d_model_traj
        self.dec_steps = args.dec_steps
        self.loc_dim = args.loc_dim
        self.dropout = args.dropout
        
        # Unified attention layer that handles both temporal alignment and cross-modality fusion
        self.unified_attention = nn.MultiheadAttention(
            embed_dim=args.d_model_traj,
            num_heads=args.n_head,
            dropout=args.dropout
        )
        
        # Simplified fusion
        self.fusion = nn.Sequential(
            nn.Linear(args.d_model_traj * 2, args.d_model_traj),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        )
        
        # Output MLP remains same
        self.output_mlp = nn.Sequential(
            nn.Linear(args.d_model_traj, args.d_model_traj * 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.d_model_traj * 2, args.loc_dim * args.dec_steps)
        )

    def forward(self, traj_features, visual_features):
        B = traj_features.shape[0]
        
        # Unified attention does both temporal alignment and cross-modality fusion
        # Query: trajectory features
        # Key/Value: visual features
        attn_output, _ = self.unified_attention(
            query=traj_features.transpose(0, 1),  # [15, B, 256]
            key=visual_features.transpose(0, 1),  # [8, B, 256]
            value=visual_features.transpose(0, 1) # [8, B, 256]
        )
        attn_output = attn_output.transpose(0, 1)  # [B, 15, 256]
        
        # Residual connection and fusion
        fused = torch.cat([traj_features, attn_output], dim=-1)
        fused = self.fusion(fused)
        
        # Final prediction
        pred = self.output_mlp(fused.mean(dim=1))  # Pool over time
        pred = pred.view(B, self.dec_steps, self.loc_dim)
        
        return pred


class Trajnet(nn.Module):
    def __init__(self, args, device):
        super(Trajnet, self).__init__()
        self.device = device
        self.dec_steps = args.dec_steps
        self.loc_dim = args.loc_dim
        
        # Trajectory and speed encoder
        self.traj_encoder = TrajEncoder(args,device)
        
        # Visual encoder (ViT-based)
        self.visual_encoder = VisualEncoder(args, device)
        
        # Decoder that combines both modalities
        self.decoder = TrajDecoder(args)
    
    def forward(self, inputs, targets=0, start_index=0, training=True, mask=None, loop=None, frames=None):
        traj, speed = inputs  # traj: (B, T, 4), speed: (B, T, 1)
        B = traj.shape[0]
        
        # Process trajectory and speed data
        traj_features, traj_hidden = self.traj_encoder(traj, speed)
        
        if frames is not None:
            # Process visual data
            visual_features, visual_hidden = self.visual_encoder(frames, traj, speed)
            
            # Decode combined features to generate prediction
            pred = self.decoder(traj_features, visual_features)
        else:
            # Fallback if no frames are provided - use only trajectory features
            dummy_visual_features = torch.zeros_like(traj_features)
            pred = self.decoder(traj_features, dummy_visual_features)
        
        return pred


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.att_enc_in = MultiHeadAttention(args, block='enc')
        self.feedforward_enc = FeedForward(args, block='enc')
        
    def forward(self, traj_input):
        traj_att = self.att_enc_in(traj_input, traj_input, traj_input, self.enc_steps, self.enc_steps)
        traj_enc = self.feedforward_enc(traj_att)    
        return traj_enc


class DecoderLayer(nn.Module):
    def __init__(self, args, block):
        super(DecoderLayer, self).__init__()

        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.hidden_size_traj = args.hidden_size_traj
        
        if block == 'traj':
            self.att_enc_out = MultiHeadAttention(args, block='enc_dec_traj')
            self.feedforward_dec = FeedForward(args, block='dec_traj')
        
    def forward(self, dec_in, enc_out, mask=None):
        enc_dec_att = self.att_enc_out(dec_in, enc_out, enc_out, self.dec_steps, self.enc_steps)
        dec_out_all = self.feedforward_dec(enc_dec_att)
        return dec_out_all