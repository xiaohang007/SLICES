"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
from flash_attn import flash_attn_qkvpacked_func
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F


class MatterGPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.sym_dim = kwargs.pop('sym_dim', 7)  # 默认为7维
        for k,v in kwargs.items():
            setattr(self, k, v)

class MatterGPT1Config(MatterGPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class FlashCausalSelfAttention(nn.Module):
    """
    A flash attention implementation of multi-head masked self-attention.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections into a single matrix
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Store dimensions
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Register causal mask buffer for inference
        num = int(bool(config.num_props))
        mask_len = config.block_size + num
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(mask_len, mask_len))
            .view(1, 1, mask_len, mask_len)
        )

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # 确保输入和权重使用相同的数据类型
        qkv_dtype = self.qkv.weight.dtype
        x = x.to(qkv_dtype)
        
        # 计算 QKV
        qkv = self.qkv(x)  # (B, T, 3 * n_embd)
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)

        # 使用 FlashAttention
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            causal=True
        )

        # 重塑输出并应用投影
        out = rearrange(out, 'b s h d -> b s (h d)')
        out = self.resid_drop(self.proj(out))
        
        return out, None

class Block(nn.Module):
    """ An unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = FlashCausalSelfAttention(config)  # Using FlashAttention version
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class MatterGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Basic embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(3, config.n_embd)
        
        # Properties embedding
        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)
        
        # Crystal system embedding - 7 dims one-hot input
        self.sym_nn = nn.Linear(7, config.n_embd)  # 7 types of Crystal systems
     
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer blocks
        self.blocks_1 = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.blocks_2 = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Add linear layer for feature fusion
        self.linear = nn.Linear(2 * config.n_embd, config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
                              
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, targets=None, prop=None, sym=None):
        b, t = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # Basic token embeddings
        token_embeddings = self.tok_emb(x)
        position_embeddings = self.pos_emb[:, :t, :]
        type_embeddings = self.type_emb(torch.ones((b,t), dtype=torch.long, device=x.device))
        x_embed = self.drop(token_embeddings + position_embeddings + type_embeddings)
        
        embed = x_embed
        combined_embed = x_embed
        
        # Path 1: Property conditioning (现在总是执行)
        x_1 = combined_embed

        type_embd = self.type_emb(torch.zeros((b, 1), dtype=torch.long, device=x.device))
        if prop.ndim == 2:
            p = self.prop_nn(prop.unsqueeze(1))
        else:
            p = self.prop_nn(prop)
        p += type_embd
        x_1 = torch.cat([p, x_1], 1)


        # Path 2: Symmetry conditioning (现在总是执行)
        x_2 = combined_embed
        type_embd_sym = self.type_emb(2 * torch.ones((b, 1), dtype=torch.long, device=x.device))
        s = self.sym_nn(sym)
        s = s.unsqueeze(1)
        s += type_embd_sym
        x_2 = torch.cat([s, x_2], 1)


        # Process through dual transformer blocks
        attn_maps = []
        
        # Process path 1
        for layer in self.blocks_1:
            x_1, attn_1 = layer(x_1)
            attn_maps.append(attn_1)
        
        # Process path 2
        for layer in self.blocks_2:
            x_2, attn_2 = layer(x_2)
            attn_maps.append(attn_2)
        
        # Combine the two paths
        x_concat = torch.cat([x_1, x_2], dim=-1)  # Concatenate along feature dimension
        x = self.linear(x_concat)  # Fuse features using linear layer
        x = self.ln_f(x)
        x = self.head(x)
        
        x = x[:, 1:, :]

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.reshape(-1, x.size(-1)), targets.view(-1))

        return x, loss, attn_maps, embed

    @torch.no_grad()
    def sample(self, x, steps, temperature=1.0, do_sample=False, top_k=None, top_p=None, prop=None, sym=None):
        """
        Take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Conditional sampling with property and crystal symmetry control.
    
        Args:
              x: Input sequence
              steps: Number of steps to sample for
              temperature: Sampling temperature
              do_sample: Whether to sample from the distribution
              top_k: Top-k sampling parameter
              top_p: Nucleus sampling parameter
              prop: Property conditioning tensor (optional)
              sym: Crystal system one-hot encoding (optional)
                    Expected to be a 7-dimensional one-hot vector representing:
                    [triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic]

        """
        #model.eval()
        
        def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
            """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
                Args:
                    logits: logits distribution shape (batch size x vocabulary size)
                    top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                    top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                        Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            """
            top_k = min(top_k, logits.size(-1))  # Safety check
            if top_k > 0:
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value
        
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
        
                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                logits[indices_to_remove] = filter_value
            return logits
        
        
        for k in range(steps):
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:] # crop context if needed

            # forward the model to get the logits for the index in the sequence
            logits, _, _, _ = self(x_cond, targets=None, prop=prop, sym=sym) # for sampling, no target
            
            #attn_maps_list.append(attn_maps) # save attention maps for visualization or analysis
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options OR using nucleus (top-p) filtering
            #if top_k is not None:
            #    v, _ = torch.topk(logits, top_k)
            #    logits[logits < v[:, [-1]]] = -float('Inf')
            logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)

                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely
            if do_sample:
                x_next = torch.multinomial(probs, num_samples=1)
            else:
                _, x_next = torch.topk(probs, k=1, dim=-1)

            # append sampled index to the running sequence and continue
            x = torch.cat((x, x_next), dim=1)

        return x[:, 1:]


