"""
Model architectures for StruCVAE-Pep.
Includes: Sequence encoder, DMPNN structure encoder, cross-attention fusion, 
disentangled VAE, and conditional decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import scatter
import math


class SequenceEncoder(nn.Module):
    """LSTM-based sequence encoder."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, 
                 hidden_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
    def forward(self, x, lengths=None):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        if lengths is not None:
            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (batch, hidden_dim * 2)
        else:
            hidden = hidden[-1]  # (batch, hidden_dim)
            
        return hidden, output


class DMPNNConv(MessagePassing):
    """Directed Message Passing Neural Network layer."""
    
    def __init__(self, hidden_dim: int, edge_dim: int = 4):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr):
        # x: (num_nodes, hidden_dim)
        # edge_index: (2, num_edges)
        # edge_attr: (num_edges, edge_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # x_j: (num_edges, hidden_dim)
        # edge_attr: (num_edges, edge_dim)
        msg = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg)
    
    def update(self, aggr_out, x):
        # aggr_out: (num_nodes, hidden_dim)
        return self.update_gru(aggr_out, x)


class StructureEncoder(nn.Module):
    """DMPNN-based structure encoder for molecular graphs."""
    
    def __init__(self, node_dim: int = 9, edge_dim: int = 4, 
                 hidden_dim: int = 256, num_layers: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            DMPNNConv(hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2  # mean + max pooling
        
    def forward(self, x, edge_index, edge_attr, batch):
        # x: (num_nodes, node_dim)
        # edge_index: (2, num_edges)
        # edge_attr: (num_edges, edge_dim)
        # batch: (num_nodes,) - batch assignment
        
        x = self.node_embedding(x)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x_new
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=-1)
        
        return x_pooled


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between sequence and structure features."""
    
    def __init__(self, seq_dim: int, struct_dim: int, 
                 hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        
        # Cross-attention: structure queries sequence
        self.cross_attn_struct = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        # Cross-attention: sequence queries structure
        self.cross_attn_seq = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        self.output_dim = hidden_dim * 2
        
    def forward(self, seq_features, struct_features):
        # seq_features: (batch, seq_dim) or (batch, seq_len, seq_dim)
        # struct_features: (batch, struct_dim)
        
        if struct_features.dim() == 2:
            struct_features = struct_features.unsqueeze(1)  # (batch, 1, struct_dim)
        if seq_features.dim() == 2:
            seq_features = seq_features.unsqueeze(1)  # (batch, 1, seq_dim)
            
        seq_proj = self.seq_proj(seq_features)  # (batch, seq_len, hidden)
        struct_proj = self.struct_proj(struct_features)  # (batch, 1, hidden)
        
        # Structure attends to sequence
        struct_enhanced, _ = self.cross_attn_struct(
            struct_proj, seq_proj, seq_proj
        )  # (batch, 1, hidden)
        
        # Sequence attends to structure
        seq_enhanced, _ = self.cross_attn_seq(
            seq_proj, struct_proj, struct_proj
        )  # (batch, seq_len, hidden)
        
        # Pool sequence features
        seq_pooled = seq_enhanced.mean(dim=1)  # (batch, hidden)
        struct_pooled = struct_enhanced.squeeze(1)  # (batch, hidden)
        
        fused = torch.cat([seq_pooled, struct_pooled], dim=-1)
        return fused


class LatentEncoder(nn.Module):
    """Encoder to disentangled latent space."""
    
    def __init__(self, input_dim: int, 
                 z_structure_dim: int = 128,
                 z_property_dim: int = 64,
                 z_sequence_dim: int = 64):
        super().__init__()
        self.z_structure_dim = z_structure_dim
        self.z_property_dim = z_property_dim
        self.z_sequence_dim = z_sequence_dim
        
        total_z = z_structure_dim + z_property_dim + z_sequence_dim
        
        # Shared base
        self.base = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Factor-specific encoders
        self.struct_mu = nn.Linear(512, z_structure_dim)
        self.struct_logvar = nn.Linear(512, z_structure_dim)
        
        self.prop_mu = nn.Linear(512, z_property_dim)
        self.prop_logvar = nn.Linear(512, z_property_dim)
        
        self.seq_mu = nn.Linear(512, z_sequence_dim)
        self.seq_logvar = nn.Linear(512, z_sequence_dim)
        
    def forward(self, x):
        h = self.base(x)
        
        # Structure factor
        z_struct_mu = self.struct_mu(h)
        z_struct_logvar = self.struct_logvar(h)
        
        # Property factor
        z_prop_mu = self.prop_mu(h)
        z_prop_logvar = self.prop_logvar(h)
        
        # Sequence factor
        z_seq_mu = self.seq_mu(h)
        z_seq_logvar = self.seq_logvar(h)
        
        return {
            'z_structure': (z_struct_mu, z_struct_logvar),
            'z_property': (z_prop_mu, z_prop_logvar),
            'z_sequence': (z_seq_mu, z_seq_logvar)
        }


class LatentDecoder(nn.Module):
    """Decoder from disentangled latent space."""
    
    def __init__(self, z_structure_dim: int = 128,
                 z_property_dim: int = 64,
                 z_sequence_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        total_z = z_structure_dim + z_property_dim + z_sequence_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(total_z, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, z_structure, z_property, z_sequence):
        z = torch.cat([z_structure, z_property, z_sequence], dim=-1)
        return self.decoder(z)


class ConditionalDecoder(nn.Module):
    """Autoregressive decoder for sequence generation with FiLM conditioning."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 256, latent_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # FiLM layers for conditioning
        self.film_gamma = nn.Linear(latent_dim, hidden_dim)
        self.film_beta = nn.Linear(latent_dim, hidden_dim)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, z, target_seq=None, max_len=50, teacher_forcing_ratio=0.5):
        batch_size = z.size(0)
        device = z.device
        
        # FiLM parameters
        gamma = self.film_gamma(z).unsqueeze(1)  # (batch, 1, hidden)
        beta = self.film_beta(z).unsqueeze(1)    # (batch, 1, hidden)
        
        if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing
            embedded = self.embedding(target_seq[:, :-1])  # Exclude last token
            lstm_out, _ = self.lstm(embedded)
            
            # Apply FiLM
            lstm_out = gamma * lstm_out + beta
            lstm_out = self.dropout(lstm_out)
            
            logits = self.output(lstm_out)
            return logits
        else:
            # Autoregressive generation
            generated = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            generated[:, 0] = 1  # <SOS> token
            
            hidden = None
            for t in range(1, max_len):
                embedded = self.embedding(generated[:, t-1:t])
                lstm_out, hidden = self.lstm(embedded, hidden)
                
                # Apply FiLM
                lstm_out = gamma * lstm_out + beta
                
                logits = self.output(lstm_out).squeeze(1)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
                generated[:, t] = next_token
                
            return generated


class PropertyPredictor(nn.Module):
    """Property prediction head for permeability."""
    
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z_property):
        return self.predictor(z_property).squeeze(-1)


class StruCVAE(nn.Module):
    """Full StruCVAE-Pep model with cross-attention fusion and disentangled latents."""
    
    def __init__(self, vocab_size: int, 
                 seq_encoder_config: dict = None,
                 struct_encoder_config: dict = None,
                 z_structure_dim: int = 128,
                 z_property_dim: int = 64,
                 z_sequence_dim: int = 64,
                 use_structure: bool = True,
                 use_cross_attention: bool = True,
                 disentangled: bool = True):
        super().__init__()
        
        self.use_structure = use_structure
        self.use_cross_attention = use_cross_attention
        self.disentangled = disentangled
        
        # Sequence encoder
        seq_config = seq_encoder_config or {}
        self.seq_encoder = SequenceEncoder(vocab_size, **seq_config)
        
        # Structure encoder (optional)
        if use_structure:
            struct_config = struct_encoder_config or {}
            self.struct_encoder = StructureEncoder(**struct_config)
            
            # Cross-attention fusion
            if use_cross_attention:
                self.fusion = CrossAttentionFusion(
                    seq_dim=self.seq_encoder.output_dim,
                    struct_dim=self.struct_encoder.output_dim
                )
                encoder_input_dim = self.fusion.output_dim
            else:
                # Late fusion: simple concatenation
                encoder_input_dim = self.seq_encoder.output_dim + self.struct_encoder.output_dim
        else:
            encoder_input_dim = self.seq_encoder.output_dim
        
        # Latent encoder
        if disentangled:
            self.latent_encoder = LatentEncoder(
                encoder_input_dim,
                z_structure_dim=z_structure_dim,
                z_property_dim=z_property_dim,
                z_sequence_dim=z_sequence_dim
            )
            latent_dim = z_structure_dim + z_property_dim + z_sequence_dim
        else:
            # Single latent space
            self.latent_mu = nn.Linear(encoder_input_dim, 256)
            self.latent_logvar = nn.Linear(encoder_input_dim, 256)
            latent_dim = 256
            z_property_dim = 256
        
        # Decoder
        self.decoder = ConditionalDecoder(
            vocab_size=vocab_size,
            latent_dim=latent_dim
        )
        
        # Property predictor
        self.property_predictor = PropertyPredictor(z_property_dim)
        
        self.z_structure_dim = z_structure_dim if disentangled else 0
        self.z_property_dim = z_property_dim
        self.z_sequence_dim = z_sequence_dim if disentangled else 0
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, seq, struct_data=None, target_seq=None):
        # Encode sequence
        seq_hidden, seq_output = self.seq_encoder(seq)
        
        if self.use_structure and struct_data is not None:
            # Encode structure
            struct_hidden = self.struct_encoder(
                struct_data.x, struct_data.edge_index,
                struct_data.edge_attr, struct_data.batch
            )
            
            if self.use_cross_attention:
                # Cross-attention fusion
                fused = self.fusion(seq_output, struct_hidden)
            else:
                # Late fusion
                fused = torch.cat([seq_hidden, struct_hidden], dim=-1)
        else:
            fused = seq_hidden
        
        # Encode to latent space
        if self.disentangled:
            latent_params = self.latent_encoder(fused)
            
            z_struct_mu, z_struct_logvar = latent_params['z_structure']
            z_prop_mu, z_prop_logvar = latent_params['z_property']
            z_seq_mu, z_seq_logvar = latent_params['z_sequence']
            
            z_structure = self.reparameterize(z_struct_mu, z_struct_logvar)
            z_property = self.reparameterize(z_prop_mu, z_prop_logvar)
            z_sequence = self.reparameterize(z_seq_mu, z_seq_logvar)
            
            z = torch.cat([z_structure, z_property, z_sequence], dim=-1)
            
            # Property prediction
            prop_pred = self.property_predictor(z_property)
            
            return {
                'z_structure': (z_struct_mu, z_struct_logvar),
                'z_property': (z_prop_mu, z_prop_logvar),
                'z_sequence': (z_seq_mu, z_seq_logvar),
                'z': z,
                'property_pred': prop_pred
            }
        else:
            # Single latent
            z_mu = self.latent_mu(fused)
            z_logvar = self.latent_logvar(fused)
            z = self.reparameterize(z_mu, z_logvar)
            
            prop_pred = self.property_predictor(z)
            
            return {
                'z': z,
                'z_mu': z_mu,
                'z_logvar': z_logvar,
                'property_pred': prop_pred
            }
    
    def generate(self, z_structure=None, z_property=None, z_sequence=None, 
                 max_len=50, batch_size=1, device='cpu'):
        """Generate sequences from latent codes."""
        if z_structure is None:
            z_structure = torch.randn(batch_size, self.z_structure_dim, device=device)
        if z_property is None:
            z_property = torch.randn(batch_size, self.z_property_dim, device=device)
        if z_sequence is None:
            z_sequence = torch.randn(batch_size, self.z_sequence_dim, device=device)
            
        z = torch.cat([z_structure, z_property, z_sequence], dim=-1)
        
        # Generate autoregressively
        generated = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        generated[:, 0] = 1  # <SOS>
        
        hidden = None
        gamma = self.decoder.film_gamma(z).unsqueeze(1)
        beta = self.decoder.film_beta(z).unsqueeze(1)
        
        for t in range(1, max_len):
            embedded = self.decoder.embedding(generated[:, t-1:t])
            lstm_out, hidden = self.decoder.lstm(embedded, hidden)
            lstm_out = gamma * lstm_out + beta
            logits = self.decoder.output(lstm_out).squeeze(1)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1)
            generated[:, t] = next_token
            
        return generated


class DMPNNPredictor(nn.Module):
    """DMPNN for direct property prediction (baseline)."""
    
    def __init__(self, node_dim: int = 9, edge_dim: int = 4,
                 hidden_dim: int = 300, num_layers: int = 5):
        super().__init__()
        self.encoder = StructureEncoder(node_dim, edge_dim, hidden_dim, num_layers)
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, edge_attr, batch):
        h = self.encoder(x, edge_index, edge_attr, batch)
        return self.predictor(h).squeeze(-1)


if __name__ == '__main__':
    # Test model
    print("Testing model architectures...")
    
    # Test sequence encoder
    seq_encoder = SequenceEncoder(vocab_size=65, embed_dim=128, hidden_dim=256)
    seq_input = torch.randint(0, 65, (4, 20))
    seq_hidden, seq_out = seq_encoder(seq_input)
    print(f"Sequence encoder output: {seq_hidden.shape}")
    
    # Test structure encoder
    struct_encoder = StructureEncoder(node_dim=9, edge_dim=4, hidden_dim=256)
    x = torch.randn(100, 9)
    edge_index = torch.randint(0, 100, (2, 200))
    edge_attr = torch.randn(200, 4)
    batch = torch.zeros(100, dtype=torch.long)
    struct_out = struct_encoder(x, edge_index, edge_attr, batch)
    print(f"Structure encoder output: {struct_out.shape}")
    
    print("All tests passed!")
