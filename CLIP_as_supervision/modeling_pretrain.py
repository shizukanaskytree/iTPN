# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# By Yunjie Tian
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# --------------------------------------------------------'
import torch
import torch.nn as nn
from functools import partial
import math

from timm.models.registry import register_model
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
import torch.utils.checkpoint as checkpoint

from modeling_finetune import _cfg, Block, PatchEmbed, PatchSplit, PatchMerge, RelativePositionBias, \
    DecoupledRelativePositionBias, ConvMlp, ConvSwiGLU, ConvMlpBlock, ConvPatchEmbed, ConvPatchMerge, ConvPatchSplit

from apex.normalization import FusedLayerNorm

from rope import *


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class iTPNForMIM(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24,
                 num_heads=8, bridge_mlp_ratio=3., mlp_ratio=4., fpn_dim=256, fpn_depth=2, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, init_values=None, attn_head_dim=None,
                 norm_layer=nn.LayerNorm, patch_norm=False, use_checkpoint=False, teacher_dim=512, num_outs=1,
                 init_std=0.02,
                 cls_token=False,
                 grad_ckpt=False,
                 stop_grad_conv1=False,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_shared_decoupled_rel_pos_bias=False,
                 rope=False,
                 convmlp=False,

                 postnorm=False,
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 xavier_normal_init=False,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_outs = num_outs
        self.num_main_blocks = depth
        self.fpn_dim = fpn_dim
        self.depth_stage1 = depth_stage1
        self.depth_stage2 = depth_stage2
        self.depth = depth
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim
        self.convmlp = convmlp

        mlvl_dims = {'4': embed_dim // 4, '8': embed_dim // 2, '16': embed_dim}
        # split image into non-overlapping patches
        if convmlp:
            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                stop_grad_conv1=stop_grad_conv1,
                norm_layer=norm_layer if patch_norm else None)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=mlvl_dims['4'],
                norm_layer=norm_layer if patch_norm else None)
        num_patches = self.patch_embed.num_patches

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim * 4))

        if use_abs_pos_emb:
            if cls_token:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(window_size=self.patch_embed.patch_shape,
                                                              num_heads=num_heads)

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.rope = None

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        self.build_blocks(
            depths=[depth_stage1, depth_stage2, depth],
            dims=mlvl_dims,
            num_heads=num_heads,
            bridge_mlp_ratio=bridge_mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            attn_head_dim=attn_head_dim,
            postnorm=postnorm,
            deepnorm=deepnorm,
            subln=subln,
            xattn=xattn,
            swiglu=swiglu,
            naiveswiglu=naiveswiglu,
            rope=self.rope,
            convmlp=convmlp,
        )

        ########################### FPN PART ###########################

        self.build_fpn_blocks(
            num_outs=num_outs,
            fpn_depth=fpn_depth,
            dims=mlvl_dims,
            fpn_dim=fpn_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            convmlp=convmlp,
        )

        ## merge the output
        self.decoder_embed = nn.ModuleList()

        self.decoder_embed.append(
            nn.ModuleList([
                norm_layer(fpn_dim),
                nn.Conv2d(fpn_dim, embed_dim, 1)]
            ))
        if self.num_outs >= 2:
            self.decoder_embed.append(
                nn.ModuleList([
                    norm_layer(fpn_dim),
                    nn.Conv2d(fpn_dim, embed_dim, 2, 2)]
                ))
        if self.num_outs >= 3:
            self.decoder_embed.append(
                nn.ModuleList([
                    norm_layer(fpn_dim),
                    nn.Conv2d(fpn_dim, embed_dim, 4, 4)]
                ))

        self.norm = norm_layer(embed_dim) if not deepnorm else nn.Identity()
        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, teacher_dim)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)

        if cls_token:
            trunc_normal_(self.cls_token, std=self.init_std)
        # trunc_normal_(self.mask_token_stage1, std=init_std)
        # trunc_normal_(self.mask_token_stage2, std=init_std)
        # trunc_normal_(self.mask_token_stage3, std=init_std)
        trunc_normal_(self.mask_token, std=init_std)
        trunc_normal_(self.lm_head.weight, std=init_std)

        if xavier_normal_init:
            self.apply(self._xavier_normal_init)
            w = self.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        else:  # ori BEiT init
            self.apply(self._init_weights)
            self.fix_init_weight()

        if postnorm:
            self._reinit_respostnorm_ln()

        if deepnorm:
            init_scale = math.pow(8.0 * depth, 0.25)
            for name, p in self.named_parameters():
                if (
                        'mlp.fc' in name
                        or 'mlp.w' in name
                        or 'attn.proj' in name
                        or 'attn.v_proj' in name
                ):
                    print('deepnorm rescale:', name, '/', init_scale)
                    p.data.div_(init_scale)

        if subln:
            init_scale = math.sqrt(math.log(depth * 2))
            for name, p in self.named_parameters():
                if (
                        'mlp.fc' in name
                        or 'mlp.w' in name
                        or 'attn.proj' in name
                        or 'attn.v_proj' in name
                ):
                    print('subln rescale:', name, 'x', init_scale)
                    p.data.mul_(init_scale)

        self.grad_ckpt = grad_ckpt
        self.stop_grad_conv1 = stop_grad_conv1

    def _reinit_respostnorm_ln(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if layer.attn is not None:
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            if layer.mlp is not None:
                if self.swiglu or self.naiveswiglu:
                    rescale(layer.mlp.w3.weight.data, layer_id + 1)
                else:
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _xavier_normal_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_blocks(self,
                     depths=[3, 3, 24],
                     dims={'4': 128 // 4, '8': 256, '16': 512},
                     num_heads=8,
                     bridge_mlp_ratio=3.,
                     qkv_bias=True,
                     qk_scale=None,
                     window_size=None,
                     drop=0.,
                     attn_drop=0.,
                     drop_path_rate=0.,
                     norm_layer=nn.LayerNorm,
                     init_values=0.,
                     attn_head_dim=None,
                     postnorm=False,
                     deepnorm=False,
                     subln=False,
                     xattn=False,
                     swiglu=False,
                     naiveswiglu=False,
                     rope=False,
                     convmlp=False,
                     ):
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, depths[0] + depths[1] + depths[2]))

        self.blocks = nn.ModuleList()

        if convmlp:
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['4'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=False,
                    naiveswiglu=False,
                    rope=rope
                ) for _ in range(depths[0])
            ])
            self.blocks.append(ConvPatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                ConvMlpBlock(
                    dim=dims['8'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=0.,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=False,
                    naiveswiglu=False,
                    rope=rope
                ) for _ in range(depths[1])
            ])
            self.blocks.append(ConvPatchMerge(dims['8'], norm_layer))
        else:
            self.blocks.extend([
                Block(
                    dim=dims['4'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                    rope=self.rope
                ) for _ in range(depths[0])
            ])
            self.blocks.append(PatchMerge(dims['4'], norm_layer))
            self.blocks.extend([
                Block(
                    dim=dims['8'],
                    num_heads=0,
                    mlp_ratio=bridge_mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=next(dpr),
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=window_size,
                    depth=depths[-1],
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                    rope=self.rope
                ) for _ in range(depths[1])
            ])
            self.blocks.append(PatchMerge(dims['8'], norm_layer))

        ######### stage 3 ########
        self.blocks.extend([
            Block(
                dim=dims['16'],
                num_heads=num_heads,
                mlp_ratio=bridge_mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=next(dpr),
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=window_size,
                attn_head_dim=attn_head_dim,
                depth=depths[-1],
                postnorm=postnorm,
                deepnorm=deepnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=rope
            ) for _ in range(depths[2])
        ])

    def build_fpn_blocks(self,
                         num_outs=3,
                         fpn_depth=2,
                         dims={'4': 128 // 4, '8': 256, '16': 512},
                         fpn_dim=256,
                         qkv_bias=True,
                         qk_scale=None,
                         mlp_ratio=4.,
                         drop=0.,
                         attn_drop=0.,
                         norm_layer=nn.LayerNorm,
                         convmlp=False,
                         ):
        if convmlp:
            if num_outs > 1:
                if dims['16'] != fpn_dim:
                    self.align_dim_16tofpn = nn.Conv2d(dims['16'], fpn_dim, 1)
                else:
                    self.align_dim_16tofpn = None
                self.fpn_modules = nn.ModuleList()
                self.fpn_modules.append(
                    ConvMlpBlock(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ))
                self.fpn_modules.append(
                    ConvMlpBlock(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ))

                self.align_dim_16to8 = nn.Conv2d(dims['8'], fpn_dim, 1)
                self.split_16to8 = ConvPatchSplit(dims['16'], fpn_dim, norm_layer)
                self.block_16to8 = nn.Sequential(
                    *[ConvMlpBlock(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ) for _ in range(fpn_depth)]
                )
            if num_outs > 2:
                self.align_dim_8to4 = nn.Conv2d(dims['4'], fpn_dim, 1)
                self.split_8to4 = ConvPatchSplit(fpn_dim, fpn_dim, norm_layer)
                self.block_8to4 = nn.Sequential(
                    *[ConvMlpBlock(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ) for _ in range(fpn_depth)]
                )
                self.fpn_modules.append(
                    ConvMlpBlock(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    )
                )
        else:
            if num_outs > 1:
                if dims['16'] != fpn_dim:
                    self.align_dim_16tofpn = nn.Linear(dims['16'], fpn_dim)
                else:
                    self.align_dim_16tofpn = None
                self.fpn_modules = nn.ModuleList()
                self.fpn_modules.append(
                    Block(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ))
                self.fpn_modules.append(
                    Block(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ))

                self.align_dim_16to8 = nn.Linear(dims['8'], fpn_dim, bias=False)
                self.split_16to8 = PatchSplit(dims['16'], fpn_dim, norm_layer)
                self.block_16to8 = nn.Sequential(
                    *[Block(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ) for _ in range(fpn_depth)]
                )
            if num_outs > 2:
                self.align_dim_8to4 = nn.Linear(dims['4'], fpn_dim, bias=False)
                self.split_8to4 = PatchSplit(fpn_dim, fpn_dim, norm_layer)
                self.block_8to4 = nn.Sequential(
                    *[Block(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    ) for _ in range(fpn_depth)]
                )
                self.fpn_modules.append(
                    Block(
                        dim=fpn_dim,
                        num_heads=0,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=0.,
                        norm_layer=norm_layer
                    )
                )

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cls_token is not None:
            return {'pos_embed', 'cls_token'}
        else:
            return {'pos_embed'}

    def get_final_patch_size(self):
        return self.patch_embed.patch_size

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        B, C, H, W = x.shape
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos, mask_token=self.mask_token)  # B*L*4*4*C
        Hp, Wp = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]

        if not self.convmlp and self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        features = []
        for i, blk in enumerate(self.blocks[:-self.num_main_blocks]):
            if isinstance(blk, PatchMerge) or isinstance(blk, ConvPatchMerge):
                features.append(x)
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        x = x.flatten(2).transpose(1, 2)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        if self.grad_ckpt:
            for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
        else:
            for i, blk in enumerate(self.blocks[-self.num_main_blocks:]):
                x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        ##########################  FPN forward  ########################

        x = x.transpose(1, 2).view(B, -1, Hp, Wp)
        outs = [x] if self.align_dim_16tofpn is None else [self.align_dim_16tofpn(x)]
        if self.num_outs >= 2:
            x = self.block_16to8(self.split_16to8(x) + self.align_dim_16to8(features[1]))
            outs.append(x)
        if self.num_outs >= 3:
            x = self.block_8to4(self.split_8to4(x) + self.align_dim_8to4(features[0]))
            outs.append(x)
        if self.num_outs > 3:
            outs = [
                out.reshape(B, Hp, Wp, *out.shape[-3:]).permute(0, 5, 1, 3, 2, 4).reshape(
                    B, -1, Hp * out.shape[-3], Wp * out.shape[-2]).contiguous()
                for out in outs]
            if self.num_outs >= 4:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))
            if self.num_outs >= 5:
                outs.insert(0, F.avg_pool2d(outs[0], kernel_size=2, stride=2))

        for i, out in enumerate(outs):
            out = self.fpn_modules[i](out)
            outs[i] = out

        return outs

    def forward(self, x, bool_masked_pos):
        outs = self.forward_features(x, bool_masked_pos=bool_masked_pos)

        feats = []
        for feat, layer in zip(outs, self.decoder_embed):
            x = layer[1](layer[0](feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
            feats.append(x)
        x = feats.pop(0)
        for i, feat in enumerate(feats):
            x = x + feats[i]

        x = self.norm(x)
        return self.lm_head(x)[bool_masked_pos]


@register_model
def itpn_tiny_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=384, depth_stage1=1, depth_stage2=1, depth=12, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=False,
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_small_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=384, depth_stage1=2, depth_stage2=2, depth=20, num_heads=6, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=False,
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_base_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=False,
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_base_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_base_ConvMlp_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=512, depth_stage1=3, depth_stage2=3, depth=24, num_heads=8, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=False,
        convmlp=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_large_2240_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_large_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(FusedLayerNorm, eps=1e-6),
        xattn=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_large_2240_patch16_256(pretrained=False, **kwargs):
    model = iTPNForMIM(
        img_size=256,
        patch_size=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12,
        bridge_mlp_ratio=3., mlp_ratio=4, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def itpn_large_xattn_fusedLN_NaiveSwiGLU_subln_RoPE_xavier_3324_patch16_256(pretrained=False, **kwargs):
    model = iTPNForMIM(
        img_size=256,
        patch_size=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(FusedLayerNorm, eps=1e-6),
        xattn=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=True,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model



@register_model
def itpn_large_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
    model = iTPNForMIM(
        patch_size=16, embed_dim=768, depth_stage1=2, depth_stage2=2, depth=40, num_heads=12, bridge_mlp_ratio=3.,
        mlp_ratio=4 * 2 / 3, qkv_bias=True, num_outs=3, fpn_dim=256, fpn_depth=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        xattn=False,
        convmlp=True,
        naiveswiglu=True,
        subln=True,
        xavier_normal_init=True,
        rope=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


