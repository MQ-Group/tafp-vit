import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.resnet import resnet26d, resnet50d, resnet101d
import numpy as np

from .layers import *


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'LV_ViT_Tiny': _cfg(),
    'LV_ViT': _cfg(),
    'LV_ViT_Medium': _cfg(crop_pct=1.0),
    'LV_ViT_Large': _cfg(crop_pct=1.0),
}

def get_block(block_type, **kargs):
    if block_type=='mha':
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type=='ffn':
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type=='tr':
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate,depth,drop_path_decay='linear'):
    if drop_path_decay=='linear':
        # linear dpr decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
    elif drop_path_decay=='fix':
        # use fixed dpr
        dpr= [drop_path_rate]*depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate)==depth
        dpr=drop_path_rate
    return dpr


class LV_ViT(nn.Module):
    """ Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """
    def __init__(self, feature_map_compssion_en=False, inter_layer_token_pruning_en=False, intra_block_row_pruning_en=False,
                 img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., drop_path_decay='linear', hybrid_backbone=None, norm_layer=nn.LayerNorm, p_emb='4_2', head_dim = None,
                 skip_lam = 1.0,order=None, mix_token=False, return_dense=False):
        super().__init__()
        self.num_classes = num_classes  # 1000 by default (for ImageNet)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.output_dim = embed_dim if num_classes==0 else num_classes  # 1000 by default (for ImageNet)
        if hybrid_backbone is not None: # hybrid_backbone is None for LV_ViT
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if p_emb=='4_2':    # lvvit_t lvvit_s lvvit_m p_emb=='4_2'
                patch_embed_fn = PatchEmbed4_2
            elif p_emb=='4_2_128':  # lvvit_l p_emb=='4_2_128'
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  # conv based patch embedding

        num_patches = self.patch_embed.num_patches  # num_patches == 16 for consistency with other models

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if order is None: # lvvit_t lvvit_s lvvit_m order is None
            dpr=get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.ModuleList([
                Block(i, feature_map_compssion_en, inter_layer_token_pruning_en, intra_block_row_pruning_en,
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(depth)])
        else:   # lvvit_l order is ['tr']*24, 'tr' is transformer block==Block
            # use given order to sequentially generate modules
            dpr=get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.ModuleList([
                get_block(order[i], feature_map_compssion_en, inter_layer_token_pruning_en, intra_block_row_pruning_en,
                    dim=embed_dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, skip_lam=skip_lam)
                for i in range(len(order))])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.return_dense=return_dense  # True for lvvit_t lvvit_s lvvit_m lvvit_l
        self.mix_token=mix_token    # True for lvvit_t lvvit_s lvvit_m lvvit_l

        if return_dense:  # True for lvvit_t lvvit_s lvvit_m lvvit_l
            self.aux_head=nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)  # weight init

        self.depth = depth
        if feature_map_compssion_en or inter_layer_token_pruning_en or intra_block_row_pruning_en:
            self.compression_en = True
        else:
            self.compression_en = False

    def get_depth(self):
        return self.depth

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight, std=.02)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                nn.init.constant_(m.group_bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head # nn.Linear(self.embed_dim, num_classes)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_embeddings(self,x): # [B=64, 3, 224, 224]
        x = self.patch_embed(x) # conv based patch embedding
        return x   # [B=64, C=240/384/512/768, 14, 14]
    def forward_tokens(self, x, first_compression_layer_idx):  # transformer blocks # [B=64, N=196, C=240/384/512/768]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # add cls token
        x = x + self.pos_embed  # add position embedding. learning pos_embed
        x = self.pos_drop(x)  # dropout position embedding
        for blk in self.blocks:  # transformer blocks
            x = blk(x, first_compression_layer_idx)
        x = self.norm(x)
        return x # [B=64, N=197, C=240/384/512/768]

    def forward_features(self,x): # model forward # [B=64, 3, 224, 224]
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x) # conv based patch embedding # [B=64, C=240/384/512/768, 14, 14]
        x = x.flatten(2).transpose(1, 2)  # flatten patch tokens # [B=64, N=196, C=240/384/512/768]
        x = self.forward_tokens(x)  # transformer blocks # [B=64, N=197, C=240/384/512/768]
        return x # [B=64, N=197, C=240/384/512/768]
    
    def forward(self, x, training, first_compression_layer_idx=24): # [B=64, 3, 224, 224]
        x = self.forward_embeddings(x) # conv based patch embedding # [B=64, C=240/384/512/768, 14, 14]

        # token level mixtoken augmentation 
        if self.mix_token and self.training: # mix_token is True for lvvit_t lvvit_s lvvit_m lvvit_l. self.training is undefined
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[2],x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0,0,0,0

        x = x.flatten(2).transpose(1, 2) # flatten patch tokens # [B=64, N=197, C=240/384/512/768]
        x = self.forward_tokens(x, first_compression_layer_idx) # transformer blocks # [B=64, N=197, C=240/384/512/768]
        x_cls = self.head(x[:,0, :]) # nn.Linear(self.embed_dim, num_classes). x[:,0] is the class token. x_cls=[B=64, num_classes=1000]

        if self.return_dense:  # True for lvvit_t lvvit_s lvvit_m lvvit_l
            x_aux = self.aux_head(x[:, 1:, :]) # nn.Linear(embed_dim, num_classes). x_aux=[B=64, num_classes=1000]
            if not training: # self.training is undefined
                # feature map compression loss
                if self.blocks[0].attn.fm_comp_en:
                    fm_comp_loss = 0.0
                    for i in range(self.depth):
                        fm_comp_loss += self.blocks[i].attn.fm_comp_loss
                    return x_cls+0.5*x_aux.max(1)[0], fm_comp_loss
                return x_cls+0.5*x_aux.max(1)[0]
            
            if self.compression_en:
                # feature map compression loss
                if self.blocks[0].attn.fm_comp_en:
                    fm_comp_loss = 0.0
                    for i in range(self.depth):
                        fm_comp_loss += self.blocks[i].attn.fm_comp_loss
                    return x_cls+0.5*x_aux.max(1)[0], fm_comp_loss
                return x_cls+0.5*x_aux.max(1)[0]

            # recover the mixed part
            if self.mix_token and training: # mix_token is True for lvvit_t lvvit_s lvvit_m lvvit_l. self.training is undefined
                x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x
                x_aux = x_aux.reshape(x_aux.shape[0],patch_h*patch_w,x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

        return x_cls

@register_model
def vit(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb=1, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model


@register_model
def lvvit(pretrained=False, **kwargs):
    model = LV_ViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model

@register_model
def lvvit_t(pretrained=False, **kwargs):
    model = LV_ViT(
        patch_size=16, embed_dim=240, depth=12, num_heads=4, mlp_ratio=3.,
        p_emb='4_2',skip_lam=1., return_dense=True,mix_token=False, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Tiny']
    return model


@register_model
def lvvit_s(pretrained=False, **kwargs):
    model = LV_ViT(
        patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=False, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT']
    return model

@register_model
def lvvit_m(pretrained=False, **kwargs):
    model = LV_ViT(
        patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
        p_emb='4_2',skip_lam=2., return_dense=True,mix_token=False, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Medium']
    return model


@register_model
def lvvit_l(pretrained=False, **kwargs):
    order = ['tr']*24 # this will override depth, can also be set as None
    model = LV_ViT(
        patch_size=16, embed_dim=768,depth=24, num_heads=12, mlp_ratio=3.,
        p_emb='4_2_128',skip_lam=3., return_dense=True,mix_token=True, order=order, **kwargs)
    model.default_cfg = default_cfgs['LV_ViT_Large']
    return model
