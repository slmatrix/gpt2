"""
@purpose   : implement GPT2
@references: minGPT and nanoGPT
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn

from dataclasses import dataclass

@dataclass
class GPTConfig:
    nContextTokens: int   = 1024
    nVocabTokens  : int   = 50257
    nBlocks       : int   = 12
    nHeads        : int   = 12
    nEmbds        : int   = 768
    dprob         : float = 0.1
    bias          : bool  = True


class Attention(nn.Module):
    def __init__(self,
                 nHeads=12,
                 nEmbdDims=768,
                 nContextTokens=1024,
                 dropProb=0.1):
        super(Attention, self).__init__()
        self.nHeads         = nHeads
        self.nEmbds         = nEmbdDims
        self.nContextTokens = nContextTokens
        assert self.nEmbds % self.nHeads == 0

        self.dense1 = nn.Linear(self.nEmbds, 3*self.nEmbds)
        self.dense2 = nn.Linear(self.nEmbds, self.nEmbds)

        self.d1Drop = dropProb
        self.d2Drop = dropProb

        #create a boolean tensor mask. diagonal and below are
        #set to False, indicating no change. above diagonal will
        #be masked (set to True) in the Forward pass. ensures
        #attention is only applied to left of sequence (current
        #and past tokens).
        mask = torch.ones(nContextTokens, nContextTokens)
        mask = torch.tril(mask)
        mask = mask == 0
        mask = mask.view(1, 1, nContextTokens, nContextTokens)
        self.mask = mask.to("cuda")


    def forward(self, X):
        p1, p2 = self.d1Drop, self.d2Drop
        nHeads = self.nHeads
        nEmbds = self.nEmbds
        bSize, nTokens, eDims = X.size()
        eDimsHead = eDims//nHeads

        #[bSize,nTokens,eDims] -> [bSize,nTokens,3*eDims]
        #the first dense layer creates three matrices: Q,K,V
        X = self.dense1(X)
        #seperate the matrices; each is [bSize,nTokens,eDims]
        Q, K, V = X.split(nEmbds, dim=2)

        #chuck matrices into smaller matrices (heads)
        #[bSize,nTokens,eDims] ->
        #[bSize,nTokens,nHeads,eDims/nHeads]
        Q = Q.view(bSize, nTokens, nHeads, eDimsHead)
        K = K.view(bSize, nTokens, nHeads, eDimsHead)
        V = V.view(bSize, nTokens, nHeads, eDimsHead)

        #[bSize,nTokens,nHeads,eDimsHead] ->
        #[bSize,nHeads,nTokens,eDimsHead]
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)
        #[bSize,nTokens,nHeads,eDimsHead] ->
        #[bSize,nHeads,nTokens,eDimsHead] ->
        #[bSize,nHeads,eDimsHead,nTokens]
        K = K.transpose(1, 2)
        K = K.transpose(2, 3)

        #SELF-ATTENTION
        #[bSize,nHeads,nTokens,eDimsHead] *
        #[bSize,nHeads,eDimsHead,nTokens] =
        #[bSize,nHeads,nTokens,nTokens]
        attn_matrix  = Q @ K  #similarity score
        attn_matrix *= 1.0/math.sqrt(eDimsHead)  #scaling

        #CAUSALITY
        #mask future tokens. this forces attention unto only
        #previously seen tokens. otherwise next token prediction
        #degenerates into output lookups. all values above the
        #diagonal are -inf.
        attn_matrix.masked_fill_(
            self.mask[:,:,:nTokens,:nTokens], -torch.inf)

        attn_matrix = fn.softmax(attn_matrix, dim=-1)
        attn_mattix = fn.dropout(attn_matrix, p=p1)

        #CONTEXT
        #[bSize,nHeads,nTokens,nTokens]   *
        #[bSize,nHeads,nTokens,eDimsHead] =
        #[bSize,nHeads,nTokens,eDimsHead]
        Z = attn_matrix @ V

        #concatenate heads
        Z = Z.transpose(1, 2)
        Z = Z.contiguous()
        Z = Z.view(bSize, nTokens, eDims)

        # output projection
        Z = self.dense2(Z)
        Z = fn.dropout(Z, p=p2)
        return Z


class MLP(nn.Module):
    def __init__(self,
                 nEmbdDims=768):
        super(MLP, self).__init__()

        self.dense1 = nn.Linear(nEmbdDims, 4*nEmbdDims)
        self.dense2 = nn.Linear(4*nEmbdDims, nEmbdDims)


    def forward(self, X):
        Y = self.dense1(X)
        Y = fn.gelu(Y)
        Y = self.dense2(Y)
        Y = fn.dropout(Y)
        return Y


class Basic(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 nHeads=12,
                 nEmbdDims=768,
                 nContextTokens=1024):
        super(Basic, self).__init__()

        self.norm1 = nn.LayerNorm(nEmbdDims)
        self.attn  = Attention(nHeads,nEmbdDims,nContextTokens)
        self.norm2 = nn.LayerNorm(nEmbdDims)
        self.mlp   = MLP(nEmbdDims)


    def forward(self, X):
        Y = self.norm1(X)
        Y = self.attn(Y)
        Y = X + Y
        
        X = Y
        Y = self.norm2(Y)
        Y = self.mlp(Y)
        Y = X + Y

        return Y


class GPT2(nn.Module):
    def __init__(self,
                 nHeads=12,
                 nBlocks=12,
                 nEmbdDims=768,
                 nContextTokens=1024,
                 nVocabTokens=50257,
                 initialize=False):
        super().__init__()
        self.nBlocks        = nBlocks
        self.nEmbdDims      = nEmbdDims
        self.nVocabTokens   = nVocabTokens
        self.nContextTokens = nContextTokens
         
        self.tEmbds = nn.Embedding(nVocabTokens,nEmbdDims)
        self.pEmbds = nn.Embedding(nContextTokens,nEmbdDims)

        self.blocks = nn.ModuleList()
        for _ in range(nBlocks):
            block = Basic(nHeads,nEmbdDims,nContextTokens)
            self.blocks.append(block)

        self.norm = nn.LayerNorm(nEmbdDims)
        self.head = nn.Linear(nEmbdDims,nVocabTokens,bias=False)

        if initialize: self.init_weights()


    def forward(self, X, Y=None):
        bSize, nTokens = X.size()
        assert nTokens <= self.nContextTokens
        positions = torch.arange(0,nTokens,device="cuda")

        #[bSize,nTokens,nEmbdDims]
        #[nTokens,nEmbdDims]
        tEmbds = self.tEmbds(X)
        pEmbds = self.pEmbds(positions)
        X = tEmbds + pEmbds
        X = fn.dropout(X)

        for block in self.blocks: X = block(X)

        X = self.norm(X)
        X = self.head(X)

        loss = None
        if Y: loss = fn.cross_entropy(X.flatten(), Y.flatten())

        return X, loss


    def init_weights(self):
        for name, module in self.named_modules():
            if   "dense1" in name:
                #default GPT2 uses bias; slower and worse perf.
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.normal_(module.weight, std=0.02)
            elif "dense2" in name:
                #see GPT2 paper. last FC layer has its own init.
                torch.nn.init.zeros_(module.bias)
                std = 0.02/math.sqrt(2*self.nBlocks)
                torch.nn.init.normal_(module.weight,std=std)
            elif isinstance(module, nn.Embedding):
                #default GPT2 uses bias; slower and worse perf.
                torch.nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    

    def crop_context_window(self, newLength):
        assert newLength <= self.contextLength
        self.contextLength = newLength
        self.pEmbds = nn.Parameter(self.pEmbds[:newLength])
        for block in self.blocks:
            block.attn.mask =\
                block.attn.mask[:,:,:newLength,:newLength]


    def configure_optimizer(self,
                            wDecay,
                            lRate,
                            betas):
        #create dict of only trainable parameters
        params  = {pn:p for pn,p in 
                   self.named_parameters() if p.requires_grad}
        #2D weights (nn.Linear, nn.Embedding) are decayed
        #1D weigths (bias, layer_norm) are NOT decayed
        decay   = [p for n,p in params.items() if p.dim()>=2]
        nodecay = [p for n,p in params.items() if p.dim()<2]
        groups  = [{"params":decay,   "weight_decay":wDecay},
                   {"params":nodecay, "weight_decay":0.0}]

        return torch.optim.AdamW(groups, lr=lRate, betas=betas)


    @classmethod
    def refactor_keys(cls, hf_model_sd):
        #refactor HF layer names
        renamed = {'wpe':'pEmbds','wte':'tEmbds', 'ln_f':'norm',
        'c_fc':'dense1','c_attn':'dense1', 'c_proj':'dense2', 
        'ln_1':'norm1','ln_2':'norm2', 'lm_head':'head'}
        for oldKey in list(hf_model_sd.keys()):
            #rename the transformer blocks
            newKey = oldKey.replace("transformer.h",  "blocks")
            newKey = newKey.replace("transformer.",  "")
            for rkey in renamed:
                if rkey in newKey:
                    newKey = newKey.replace(rkey,renamed[rkey])
            #renamed: oldKey -> newKey
            hf_model_sd[newKey] = hf_model_sd.pop(oldKey)


    @classmethod
    def load_weights(cls, name):
        r"""
        initialize GPT2 from pretrained weights.
        """
        import re
        from transformers import GPT2LMHeadModel
        assert name in ["gpt2"      , "gpt2-medium",\
                        "gpt2-large", "gpt2-xl"]

        params = {
        'gpt2'       :dict(nBlocks=12,nHeads=12,nEmbdDims=768),
        'gpt2-medium':dict(nBlocks=24,nHeads=16,nEmbdDims=1024),
        'gpt2-large' :dict(nBlocks=36,nHeads=20,nEmbdDims=1280),
        'gpt2-xl'    :dict(nBlocks=48,nHeads=25,nEmbdDims=1600)}

        #instantiate our untrained GPT2 model
        #also load in pretrained huggingface model
        my_model = GPT2(**params[name])
        hf_model = GPT2LMHeadModel.from_pretrained(name)
        my_model_sd = my_model.state_dict()
        hf_model_sd = hf_model.state_dict()
        assert len(my_model_sd) == len(hf_model_sd)
        cls.refactor_keys(hf_model_sd)

        #load HF weights into my model definition
        for key in hf_model_sd.keys():
            #transpose the Conv1D weights; we use nn.Linear
            if "dense" in key: hf_model_sd[key].t_()
            my_model_sd[key].copy_(hf_model_sd[key])

        return my_model


    @torch.inference_mode()
    def generate(self,
                 outTokens,
                 nNewTokens,
                 temperature=1.0,
                 topk=None):
        for _ in range(nNewTokens):
            #crop context; long generation
            if outTokens.shape[1] > self.nContextTokens:
                context = outTokens[:,-self.nContextTokens:]
            else:
                context = outTokens

            #
            logits, _ = self.forward(context)
            logits    = logits[:,-1,:]
            logits   /= temperature

            #optional mask of logits below topk
            if topk:
                topk    = min(topk, logits.size(-1))
                mask, _ = torch.topk(logits, min(topk,logits.shape[-1]))
                logits[logits < mask[:,[-1]]] = -torch.inf

            #normalize, then sample a logit probability
            probs    = fn.softmax(logits, dim=-1)
            newToken = torch.multinomial(probs,num_samples=1)

            outTokens = torch.cat((outTokens, newToken), dim=1)

        return outTokens
