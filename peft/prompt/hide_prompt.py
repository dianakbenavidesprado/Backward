import torch
import torch.nn as nn


class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)  # TODO fix self.num_layers = 1
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            #prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            #self.prompt_key = prompt_mean 
            #DBP changes
            with torch.no_grad():
                # Average over Layers(0), Dual(1), Length(3), and Heads(4)
                prompt_mean = torch.mean(self.prompt, dim=[0, 1, 3, 4]) 
                # Reshape to [pool_size, embed_dim]
                self.prompt_key = prompt_mean.reshape(self.pool_size, -1)
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, prompt_idx=None, prompt_weight=None, prompt_momentum=0):
        assert prompt_mask is not None or prompt_idx is not None or prompt_weight is not None
        assert self.prompt_pool, "In HiDe-Prompt, 'prompt_pool' must be set to True"
        out = dict()
        if self.prompt_pool:
            idx = prompt_idx

            if self.batchwise_prompt and prompt_idx is not None:
                prompt_id, id_counts = torch.unique(prompt_idx, return_counts=True, sorted=True)
                
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(prompt_idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k
            
            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k
            if idx is not None:
                out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                
                # 1. Identify current and past indices
                curr_idx = idx[0][0].item() if idx is not None else 0
                        
                if prompt_weight is not None:
                    batched_prompt_raw = torch.einsum("bp,ndplhe->ndblhe", prompt_weight, self.prompt) # num_layers, 2, B, top_k, length, C
                    batched_prompt_raw = batched_prompt_raw.unsqueeze(3)
                    #num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    # print(top_k)
                    #batched_prompt = batched_prompt_raw.reshape(
                        #num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    #)
                elif prompt_momentum > 0 and prompt_mask is not None:
                    with torch.no_grad():
                    #DBP begin adaptive prompt momentum
                        
                        device = self.prompt.device
    
                        # Ensure keys are the right shape for cosine_similarity [1, D] and [K, D]
                        curr_key = self.prompt_key[curr_idx].view(1, -1).to(device)
                        past_keys = self.prompt_key[:curr_idx].view(curr_idx, -1).to(device)

                        temp = 0.1 
                        sim = torch.nn.functional.cosine_similarity(curr_key, past_keys, dim=1)
    
                        # Force weights to be 1D [curr_idx] to match the 'p' in einsum
                        weights = torch.softmax(sim / temp, dim=0).view(-1) 

                        # Identify previous prompts
                        past_prompts = self.prompt[:, :, :curr_idx].detach().clone()
    
                        # Perform the weighted sum (ensemble)
                        # weights(p) * past_prompts(n, d, p, l, h, e) -> (n, d, l, h, e)
                        batched_prompt_momentum = torch.einsum('p,ndplhe->ndlhe', weights, past_prompts)
    
                        # Add the pool dimension back to match the original prompt shape [n, d, 1, l, h, e]
                        batched_prompt_momentum = batched_prompt_momentum.unsqueeze(2).unsqueeze(3)                    
                   
                    batched_prompt_raw = (1 - prompt_momentum) * self.prompt[:, :, idx] + prompt_momentum * batched_prompt_momentum
                    
                else:
                    batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, B, top_k, length, C
                    
                
                if self.training:
                    # Dropout starts high for Task 0 and decreases for later tasks
                    initial_drop = 0.7  # Start by dropping 70% of tokens for Task 1
                    drop_step = 0.1    # Decrease drop rate by 10% every task
                    current_drop_rate = max(0.0, initial_drop - (drop_step * curr_idx))

                    if current_drop_rate > 0:
                        # 1. Create a "seed" from the input data to simulate randomness
                        # We take the mean of x_embed and use its sine wave as a pseudo-random base
                        # Shape of x_embed is usually [B, N, C]
                        with torch.no_grad():
                            seed = x_embed.detach().mean() 
                            # Create a deterministic but "messy" sequence of numbers for the length
                            # We use a large prime multiplier and sine to spread the values
                            indices = torch.arange(self.length, device=x_embed.device).float()
                            pseudo_random = torch.sin(indices * 12.9898 + seed * 78.233) * 43758.5453
                            pseudo_random = pseudo_random - torch.floor(pseudo_random) # Get decimals [0, 1]
                        
                        # 2. Reshape to match [1, 1, 1, 1, Length, 1, 1]
                        pseudo_random = pseudo_random.view(1, 1, 1, 1, self.length, 1, 1)
                        
                        # 3. Create keep_mask (1 = keep, 0 = drop)
                        keep_prob = 1.0 - current_drop_rate
                        keep_mask = (pseudo_random < keep_prob).to(batched_prompt_raw.dtype)
                        
                        # 4. Apply and scale
                        batched_prompt_raw = (batched_prompt_raw * keep_mask) / keep_prob              
                    
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim)
                #DBP end
            else:
                if prompt_weight is not None:
                    batched_prompt_raw = torch.einsum("bp,npld->nbpld", prompt_weight, self.prompt)
                    num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )
                else:
                    batched_prompt_raw = self.prompt[:, idx]
                    num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                    batched_prompt = batched_prompt_raw.reshape(
                        num_layers, batch_size, top_k * length, embed_dim
                    )
        
        out['batched_prompt'] = batched_prompt

        return out
