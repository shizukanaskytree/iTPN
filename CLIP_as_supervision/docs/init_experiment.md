called `CLIP_as_supervision/run_scripts/run.sh`
commit id: `0aec28c2ae66e1de7bd2525699338141dacc4d98`

```bash
(itpn-py310) (base) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision$ bash run_scripts/run.sh
The running command is: torchrun --nnodes=1 --nproc_per_node=4         --node_rank=0 --master_addr=localhost --master_port=8888         run_itpn_pretraining.py         --output_dir=/tmp/output/         --log_dir=/tmp/output          --model itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224          --teacher_type clip         --clip_model EVA_CLIP_g_14_X         --cache_dir /tmp/eva_clip_psz14.pt         --input_size 224         --second_input_size 224          --second_interpolation 'bicubic'          --num_mask_patches 75         --layer_scale_init_value 0.1          --batch_size 32         --lr 0.0015          --epochs 300         --warmup_epochs 10          --teacher_dim 768         --clip_path ../ViT-B-16.pt         --clip_grad 3.0          --drop_path 0.1          --imagenet_default_mean_and_std          --opt_betas 0.9 0.98          --min_crop_scale 0.2         --opt_eps 1e-08           --save_ckpt_freq 10         --zero_stage 1         --update_freq 1         --stop_grad_conv1         --enable_deepspeed
W0724 02:13:08.958000 140694527051584 torch/distributed/run.py:757]
W0724 02:13:08.958000 140694527051584 torch/distributed/run.py:757] *****************************************
W0724 02:13:08.958000 140694527051584 torch/distributed/run.py:757] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0724 02:13:08.958000 140694527051584 torch/distributed/run.py:757] *****************************************
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:726: UserWarning: Overwriting itpn_base_3324_patch16_224 in registry with modeling_pretrain.itpn_base_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:741: UserWarning: Overwriting itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224 in registry with modeling_pretrain.itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:806: UserWarning: Overwriting itpn_large_2240_patch16_224 in registry with modeling_pretrain.itpn_large_2240_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:842: UserWarning: Overwriting itpn_large_2240_patch16_256 in registry with modeling_pretrain.itpn_large_2240_patch16_256. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_256(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:726: UserWarning: Overwriting itpn_base_3324_patch16_224 in registry with modeling_pretrain.itpn_base_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:741: UserWarning: Overwriting itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224 in registry with modeling_pretrain.itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:806: UserWarning: Overwriting itpn_large_2240_patch16_224 in registry with modeling_pretrain.itpn_large_2240_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:842: UserWarning: Overwriting itpn_large_2240_patch16_256 in registry with modeling_pretrain.itpn_large_2240_patch16_256. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_256(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:726: UserWarning: Overwriting itpn_base_3324_patch16_224 in registry with modeling_pretrain.itpn_base_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:741: UserWarning: Overwriting itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224 in registry with modeling_pretrain.itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:806: UserWarning: Overwriting itpn_large_2240_patch16_224 in registry with modeling_pretrain.itpn_large_2240_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:842: UserWarning: Overwriting itpn_large_2240_patch16_256 in registry with modeling_pretrain.itpn_large_2240_patch16_256. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_256(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:726: UserWarning: Overwriting itpn_base_3324_patch16_224 in registry with modeling_pretrain.itpn_base_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:741: UserWarning: Overwriting itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224 in registry with modeling_pretrain.itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:806: UserWarning: Overwriting itpn_large_2240_patch16_224 in registry with modeling_pretrain.itpn_large_2240_patch16_224. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_224(pretrained=False, **kwargs):
/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision/modeling_pretrain.py:842: UserWarning: Overwriting itpn_large_2240_patch16_256 in registry with modeling_pretrain.itpn_large_2240_patch16_256. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
  def itpn_large_2240_patch16_256(pretrained=False, **kwargs):
[2024-07-24 02:13:11,390] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-24 02:13:11,433] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
[2024-07-24 02:13:11,472] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-07-24 02:13:11,479] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.1), only 1.0.0 is known to be compatible
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.1), only 1.0.0 is known to be compatible
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.1), only 1.0.0 is known to be compatible
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.1), only 1.0.0 is known to be compatible
| distributed init (rank 0): env://, gpu 0
| distributed init (rank 1): env://, gpu 1
| distributed init (rank 3): env://, gpu 3
| distributed init (rank 2): env://, gpu 2
Namespace(batch_size=32,
epochs=300,
update_freq=1,
save_ckpt_freq=10,
clip_path='../ViT-B-16.pt',
teacher_dim=768,
teacher_type='clip',
teacher_model_path=None,
clip_model='EVA_CLIP_g_14_X',
cache_dir='/tmp/eva_clip_psz14.pt',
model='itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224',
grad_ckpt=False,
stop_grad_conv1=True,
rel_pos_bias=False,
decoupled_rel_pos_bias=False,
abs_pos_emb=True,
layer_scale_init_value=0.1,
load_tar=False,
num_mask_patches=75,
max_mask_patches_per_block=None,
min_mask_patches_per_block=16,
input_size=224,
second_input_size=224,
drop_path=0.1,
opt='adamw',
opt_eps=1e-08,
opt_betas=[0.9,
0.98],
clip_grad=3.0,
momentum=0.9,
weight_decay=0.05,
weight_decay_end=None,
lr=0.0015,
warmup_lr=1e-06,
min_lr=1e-05,
warmup_epochs=10,
warmup_steps=-1,
color_jitter=0.0,
train_interpolation='bicubic',
second_interpolation='bicubic',
crop_scale=[0.2,
1.0],
crop_ratio=[0.75,
1.3333333333333333],
min_crop_scale=0.2,
data_path='/datasets01/imagenet_full_size/061417/',
eval_data_path='IMNET',
data_set='image_folder',
imagenet_default_mean_and_std=True,
output_dir='/tmp/output/',
log_dir='/tmp/output',
device='cuda',
seed=0,
resume='',
auto_resume=True,
start_epoch=0,
num_workers=10,
pin_mem=True,
world_size=4,
local_rank=-1,
dist_on_itp=False,
dist_url='env://',
bf16=False,
enable_deepspeed=True,
zero_stage=1,
deepspeed=False,
deepspeed_config='/tmp/output/deepspeed_config.json',
deepscale=False,
deepscale_config=None,
rank=0,
gpu=0,
distributed=True,
dist_backend='nccl')
Creating clip teacher:
Creating clip teacher: ViT-B/16
teacher = clip_distill(
  (scaling_layer): ScalingLayerForClip()
  (teacher_model): CLIP(
    (visual): VisionTransformer(
      (conv1): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)
      (ln_pre): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (transformer): Transformer(
        (resblocks): Sequential(
          (0): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (1): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (2): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (3): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (4): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (5): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (6): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (7): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (8): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (9): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (10): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
          (11): ResidualAttentionBlock(
            (attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
            )
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=768, out_features=3072, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=3072, out_features=768, bias=True)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (ln_post): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (transformer): Transformer(
      (resblocks): Sequential(
        (0): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (1): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (2): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (3): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (4): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (5): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (6): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (7): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (8): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (9): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (10): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
        (11): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (ln_1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=512, out_features=2048, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=2048, out_features=512, bias=True)
          )
          (ln_2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (token_embedding): Embedding(49408, 512)
    (ln_final): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (LN): LayerNorm((512,), eps=1e-05, elementwise_affine=False)
)
Creating model: itpn_base_fusedLN_NaiveSwiGLU_subln_xavier_3324_patch16_224
subln rescale: blocks.0.mlp.fc1.weight x 1.9675367876885788
subln rescale: blocks.0.mlp.fc1.bias x 1.9675367876885788
subln rescale: blocks.0.mlp.fc2.weight x 1.9675367876885788
subln rescale: blocks.0.mlp.fc2.bias x 1.9675367876885788
subln rescale: blocks.1.mlp.fc1.weight x 1.9675367876885788
subln rescale: blocks.1.mlp.fc1.bias x 1.9675367876885788
subln rescale: blocks.1.mlp.fc2.weight x 1.9675367876885788
subln rescale: blocks.1.mlp.fc2.bias x 1.9675367876885788
subln rescale: blocks.2.mlp.fc1.weight x 1.9675367876885788
subln rescale: blocks.2.mlp.fc1.bias x 1.9675367876885788
subln rescale: blocks.2.mlp.fc2.weight x 1.9675367876885788
subln rescale: blocks.2.mlp.fc2.bias x 1.9675367876885788
subln rescale: blocks.4.mlp.fc1.weight x 1.9675367876885788
subln rescale: blocks.4.mlp.fc1.bias x 1.9675367876885788
subln rescale: blocks.4.mlp.fc2.weight x 1.9675367876885788
subln rescale: blocks.4.mlp.fc2.bias x 1.9675367876885788
subln rescale: blocks.5.mlp.fc1.weight x 1.9675367876885788
subln rescale: blocks.5.mlp.fc1.bias x 1.9675367876885788
subln rescale: blocks.5.mlp.fc2.weight x 1.9675367876885788
subln rescale: blocks.5.mlp.fc2.bias x 1.9675367876885788
subln rescale: blocks.6.mlp.fc1.weight x 1.9675367876885788
subln rescale: blocks.6.mlp.fc1.bias x 1.9675367876885788
subln rescale: blocks.6.mlp.fc2.weight x 1.9675367876885788
subln rescale: blocks.6.mlp.fc2.bias x 1.9675367876885788
subln rescale: blocks.8.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.8.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.8.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.8.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.8.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.8.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.8.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.8.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.8.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.9.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.9.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.9.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.9.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.9.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.9.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.9.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.9.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.9.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.10.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.10.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.10.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.10.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.10.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.10.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.10.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.10.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.10.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.11.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.11.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.11.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.11.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.11.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.11.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.11.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.11.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.11.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.12.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.12.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.12.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.12.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.12.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.12.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.12.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.12.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.12.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.13.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.13.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.13.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.13.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.13.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.13.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.13.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.13.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.13.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.14.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.14.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.14.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.14.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.14.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.14.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.14.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.14.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.14.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.15.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.15.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.15.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.15.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.15.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.15.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.15.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.15.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.15.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.16.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.16.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.16.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.16.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.16.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.16.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.16.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.16.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.16.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.17.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.17.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.17.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.17.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.17.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.17.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.17.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.17.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.17.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.18.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.18.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.18.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.18.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.18.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.18.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.18.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.18.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.18.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.19.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.19.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.19.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.19.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.19.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.19.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.19.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.19.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.19.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.20.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.20.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.20.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.20.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.20.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.20.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.20.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.20.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.20.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.21.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.21.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.21.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.21.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.21.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.21.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.21.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.21.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.21.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.22.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.22.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.22.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.22.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.22.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.22.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.22.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.22.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.22.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.23.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.23.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.23.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.23.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.23.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.23.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.23.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.23.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.23.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.24.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.24.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.24.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.24.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.24.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.24.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.24.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.24.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.24.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.25.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.25.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.25.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.25.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.25.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.25.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.25.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.25.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.25.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.26.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.26.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.26.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.26.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.26.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.26.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.26.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.26.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.26.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.27.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.27.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.27.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.27.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.27.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.27.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.27.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.27.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.27.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.28.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.28.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.28.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.28.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.28.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.28.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.28.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.28.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.28.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.29.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.29.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.29.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.29.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.29.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.29.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.29.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.29.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.29.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.30.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.30.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.30.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.30.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.30.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.30.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.30.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.30.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.30.mlp.w3.bias x 1.9675367876885788
subln rescale: blocks.31.attn.v_proj.weight x 1.9675367876885788
subln rescale: blocks.31.attn.proj.weight x 1.9675367876885788
subln rescale: blocks.31.attn.proj.bias x 1.9675367876885788
subln rescale: blocks.31.mlp.w1.weight x 1.9675367876885788
subln rescale: blocks.31.mlp.w1.bias x 1.9675367876885788
subln rescale: blocks.31.mlp.w2.weight x 1.9675367876885788
subln rescale: blocks.31.mlp.w2.bias x 1.9675367876885788
subln rescale: blocks.31.mlp.w3.weight x 1.9675367876885788
subln rescale: blocks.31.mlp.w3.bias x 1.9675367876885788
subln rescale: fpn_modules.0.mlp.fc1.weight x 1.9675367876885788
subln rescale: fpn_modules.0.mlp.fc1.bias x 1.9675367876885788
subln rescale: fpn_modules.0.mlp.fc2.weight x 1.9675367876885788
subln rescale: fpn_modules.0.mlp.fc2.bias x 1.9675367876885788
subln rescale: fpn_modules.1.mlp.fc1.weight x 1.9675367876885788
subln rescale: fpn_modules.1.mlp.fc1.bias x 1.9675367876885788
subln rescale: fpn_modules.1.mlp.fc2.weight x 1.9675367876885788
subln rescale: fpn_modules.1.mlp.fc2.bias x 1.9675367876885788
subln rescale: fpn_modules.2.mlp.fc1.weight x 1.9675367876885788
subln rescale: fpn_modules.2.mlp.fc1.bias x 1.9675367876885788
subln rescale: fpn_modules.2.mlp.fc2.weight x 1.9675367876885788
subln rescale: fpn_modules.2.mlp.fc2.bias x 1.9675367876885788
subln rescale: block_16to8.0.mlp.fc1.weight x 1.9675367876885788
subln rescale: block_16to8.0.mlp.fc1.bias x 1.9675367876885788
subln rescale: block_16to8.0.mlp.fc2.weight x 1.9675367876885788
subln rescale: block_16to8.0.mlp.fc2.bias x 1.9675367876885788
subln rescale: block_8to4.0.mlp.fc1.weight x 1.9675367876885788
subln rescale: block_8to4.0.mlp.fc1.bias x 1.9675367876885788
subln rescale: block_8to4.0.mlp.fc2.weight x 1.9675367876885788
subln rescale: block_8to4.0.mlp.fc2.bias x 1.9675367876885788
Patch size = (16, 16)
Data Aug = (DataAugmentationForiTPN,
  common_transform = Compose(
    RandomHorizontalFlip(p=0.5)
    RandomResizedCropAndInterpolationWithTwoPic(size=(224, 224), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC, second_size=(224, 224), second_interpolation=PIL.Image.BICUBIC)
),
  patch_transform = Compose(
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
),
  visual_tokens_transform = Compose(
    ToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
),
  Masked position generator = Generator(14, 14 -> [16 ~ 75], max = 75, -1.204 ~ 1.204),
)
Loading dataset shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 257/257 [00:00<00:00, 17961.41it/s]
[2024-07-24 02:13:15,966] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.4, git-hash=unknown, git-branch=unknown
[2024-07-24 02:13:15,966] [INFO] [comm.py:637:init_distributed] cdb=None
Loading dataset shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 257/257 [00:00<00:00, 21314.46it/s]
Loading dataset shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 257/257 [00:00<00:00, 15587.25it/s]
Debug: len(dataset_train): 1281167
Debug: args.batch_size: 32
Debug: num_tasks: 4
Debug: args.update_freq: 1
Debug: num_training_steps_per_epoch: 10009
Sampler_train = <torch.utils.data.distributed.DistributedSampler object at 0x7f59cc0a37f0>
LR = 0.00150000
Batch size = 128
Number of training steps = 10009
Number of training examples per epoch = 1281152
Param groups = {
  "decay": {
    "weight_decay": 0.05,
    "params": [
      "mask_token",
      "patch_embed.proj.weight",
      "blocks.0.mlp.fc1.weight",
      "blocks.0.mlp.fc2.weight",
      "blocks.1.mlp.fc1.weight",
      "blocks.1.mlp.fc2.weight",
      "blocks.2.mlp.fc1.weight",
      "blocks.2.mlp.fc2.weight",
      "blocks.3.reduction.weight",
      "blocks.4.mlp.fc1.weight",
      "blocks.4.mlp.fc2.weight",
      "blocks.5.mlp.fc1.weight",
      "blocks.5.mlp.fc2.weight",
      "blocks.6.mlp.fc1.weight",
      "blocks.6.mlp.fc2.weight",
      "blocks.7.reduction.weight",
      "blocks.8.attn.q_proj.weight",
      "blocks.8.attn.k_proj.weight",
      "blocks.8.attn.v_proj.weight",
      "blocks.8.attn.proj.weight",
      "blocks.8.mlp.w1.weight",
      "blocks.8.mlp.w2.weight",
      "blocks.8.mlp.w3.weight",
      "blocks.9.attn.q_proj.weight",
      "blocks.9.attn.k_proj.weight",
      "blocks.9.attn.v_proj.weight",
      "blocks.9.attn.proj.weight",
      "blocks.9.mlp.w1.weight",
      "blocks.9.mlp.w2.weight",
      "blocks.9.mlp.w3.weight",
      "blocks.10.attn.q_proj.weight",
      "blocks.10.attn.k_proj.weight",
      "blocks.10.attn.v_proj.weight",
      "blocks.10.attn.proj.weight",
      "blocks.10.mlp.w1.weight",
      "blocks.10.mlp.w2.weight",
      "blocks.10.mlp.w3.weight",
      "blocks.11.attn.q_proj.weight",
      "blocks.11.attn.k_proj.weight",
      "blocks.11.attn.v_proj.weight",
      "blocks.11.attn.proj.weight",
      "blocks.11.mlp.w1.weight",
      "blocks.11.mlp.w2.weight",
      "blocks.11.mlp.w3.weight",
      "blocks.12.attn.q_proj.weight",
      "blocks.12.attn.k_proj.weight",
      "blocks.12.attn.v_proj.weight",
      "blocks.12.attn.proj.weight",
      "blocks.12.mlp.w1.weight",
      "blocks.12.mlp.w2.weight",
      "blocks.12.mlp.w3.weight",
      "blocks.13.attn.q_proj.weight",
      "blocks.13.attn.k_proj.weight",
      "blocks.13.attn.v_proj.weight",
      "blocks.13.attn.proj.weight",
      "blocks.13.mlp.w1.weight",
      "blocks.13.mlp.w2.weight",
      "blocks.13.mlp.w3.weight",
      "blocks.14.attn.q_proj.weight",
      "blocks.14.attn.k_proj.weight",
      "blocks.14.attn.v_proj.weight",
      "blocks.14.attn.proj.weight",
      "blocks.14.mlp.w1.weight",
      "blocks.14.mlp.w2.weight",
      "blocks.14.mlp.w3.weight",
      "blocks.15.attn.q_proj.weight",
      "blocks.15.attn.k_proj.weight",
      "blocks.15.attn.v_proj.weight",
      "blocks.15.attn.proj.weight",
      "blocks.15.mlp.w1.weight",
      "blocks.15.mlp.w2.weight",
      "blocks.15.mlp.w3.weight",
      "blocks.16.attn.q_proj.weight",
      "blocks.16.attn.k_proj.weight",
      "blocks.16.attn.v_proj.weight",
      "blocks.16.attn.proj.weight",
      "blocks.16.mlp.w1.weight",
      "blocks.16.mlp.w2.weight",
      "blocks.16.mlp.w3.weight",
      "blocks.17.attn.q_proj.weight",
      "blocks.17.attn.k_proj.weight",
      "blocks.17.attn.v_proj.weight",
      "blocks.17.attn.proj.weight",
      "blocks.17.mlp.w1.weight",
      "blocks.17.mlp.w2.weight",
      "blocks.17.mlp.w3.weight",
      "blocks.18.attn.q_proj.weight",
      "blocks.18.attn.k_proj.weight",
      "blocks.18.attn.v_proj.weight",
      "blocks.18.attn.proj.weight",
      "blocks.18.mlp.w1.weight",
      "blocks.18.mlp.w2.weight",
      "blocks.18.mlp.w3.weight",
      "blocks.19.attn.q_proj.weight",
      "blocks.19.attn.k_proj.weight",
      "blocks.19.attn.v_proj.weight",
      "blocks.19.attn.proj.weight",
      "blocks.19.mlp.w1.weight",
      "blocks.19.mlp.w2.weight",
      "blocks.19.mlp.w3.weight",
      "blocks.20.attn.q_proj.weight",
      "blocks.20.attn.k_proj.weight",
      "blocks.20.attn.v_proj.weight",
      "blocks.20.attn.proj.weight",
      "blocks.20.mlp.w1.weight",
      "blocks.20.mlp.w2.weight",
      "blocks.20.mlp.w3.weight",
      "blocks.21.attn.q_proj.weight",
      "blocks.21.attn.k_proj.weight",
      "blocks.21.attn.v_proj.weight",
      "blocks.21.attn.proj.weight",
      "blocks.21.mlp.w1.weight",
      "blocks.21.mlp.w2.weight",
      "blocks.21.mlp.w3.weight",
      "blocks.22.attn.q_proj.weight",
      "blocks.22.attn.k_proj.weight",
      "blocks.22.attn.v_proj.weight",
      "blocks.22.attn.proj.weight",
      "blocks.22.mlp.w1.weight",
      "blocks.22.mlp.w2.weight",
      "blocks.22.mlp.w3.weight",
      "blocks.23.attn.q_proj.weight",
      "blocks.23.attn.k_proj.weight",
      "blocks.23.attn.v_proj.weight",
      "blocks.23.attn.proj.weight",
      "blocks.23.mlp.w1.weight",
      "blocks.23.mlp.w2.weight",
      "blocks.23.mlp.w3.weight",
      "blocks.24.attn.q_proj.weight",
      "blocks.24.attn.k_proj.weight",
      "blocks.24.attn.v_proj.weight",
      "blocks.24.attn.proj.weight",
      "blocks.24.mlp.w1.weight",
      "blocks.24.mlp.w2.weight",
      "blocks.24.mlp.w3.weight",
      "blocks.25.attn.q_proj.weight",
      "blocks.25.attn.k_proj.weight",
      "blocks.25.attn.v_proj.weight",
      "blocks.25.attn.proj.weight",
      "blocks.25.mlp.w1.weight",
      "blocks.25.mlp.w2.weight",
      "blocks.25.mlp.w3.weight",
      "blocks.26.attn.q_proj.weight",
      "blocks.26.attn.k_proj.weight",
      "blocks.26.attn.v_proj.weight",
      "blocks.26.attn.proj.weight",
      "blocks.26.mlp.w1.weight",
      "blocks.26.mlp.w2.weight",
      "blocks.26.mlp.w3.weight",
      "blocks.27.attn.q_proj.weight",
      "blocks.27.attn.k_proj.weight",
      "blocks.27.attn.v_proj.weight",
      "blocks.27.attn.proj.weight",
      "blocks.27.mlp.w1.weight",
      "blocks.27.mlp.w2.weight",
      "blocks.27.mlp.w3.weight",
      "blocks.28.attn.q_proj.weight",
      "blocks.28.attn.k_proj.weight",
      "blocks.28.attn.v_proj.weight",
      "blocks.28.attn.proj.weight",
      "blocks.28.mlp.w1.weight",
      "blocks.28.mlp.w2.weight",
      "blocks.28.mlp.w3.weight",
      "blocks.29.attn.q_proj.weight",
      "blocks.29.attn.k_proj.weight",
      "blocks.29.attn.v_proj.weight",
      "blocks.29.attn.proj.weight",
      "blocks.29.mlp.w1.weight",
      "blocks.29.mlp.w2.weight",
      "blocks.29.mlp.w3.weight",
      "blocks.30.attn.q_proj.weight",
      "blocks.30.attn.k_proj.weight",
      "blocks.30.attn.v_proj.weight",
      "blocks.30.attn.proj.weight",
      "blocks.30.mlp.w1.weight",
      "blocks.30.mlp.w2.weight",
      "blocks.30.mlp.w3.weight",
      "blocks.31.attn.q_proj.weight",
      "blocks.31.attn.k_proj.weight",
      "blocks.31.attn.v_proj.weight",
      "blocks.31.attn.proj.weight",
      "blocks.31.mlp.w1.weight",
      "blocks.31.mlp.w2.weight",
      "blocks.31.mlp.w3.weight",
      "align_dim_16tofpn.weight",
      "fpn_modules.0.mlp.fc1.weight",
      "fpn_modules.0.mlp.fc2.weight",
      "fpn_modules.1.mlp.fc1.weight",
      "fpn_modules.1.mlp.fc2.weight",
      "fpn_modules.2.mlp.fc1.weight",
      "fpn_modules.2.mlp.fc2.weight",
      "align_dim_16to8.weight",
      "split_16to8.reduction.weight",
      "block_16to8.0.mlp.fc1.weight",
      "block_16to8.0.mlp.fc2.weight",
      "align_dim_8to4.weight",
      "split_8to4.reduction.weight",
      "block_8to4.0.mlp.fc1.weight",
      "block_8to4.0.mlp.fc2.weight",
      "decoder_embed.0.1.weight",
      "decoder_embed.1.1.weight",
      "decoder_embed.2.1.weight",
      "lm_head.weight"
    ],
    "lr_scale": 1.0
  },
  "no_decay": {
    "weight_decay": 0.0,
    "params": [
      "pos_embed",
      "patch_embed.proj.bias",
      "blocks.0.norm2.weight",
      "blocks.0.norm2.bias",
      "blocks.0.mlp.fc1.bias",
      "blocks.0.mlp.ffn_ln.weight",
      "blocks.0.mlp.ffn_ln.bias",
      "blocks.0.mlp.fc2.bias",
      "blocks.1.norm2.weight",
      "blocks.1.norm2.bias",
      "blocks.1.mlp.fc1.bias",
      "blocks.1.mlp.ffn_ln.weight",
      "blocks.1.mlp.ffn_ln.bias",
      "blocks.1.mlp.fc2.bias",
      "blocks.2.norm2.weight",
      "blocks.2.norm2.bias",
      "blocks.2.mlp.fc1.bias",
      "blocks.2.mlp.ffn_ln.weight",
      "blocks.2.mlp.ffn_ln.bias",
      "blocks.2.mlp.fc2.bias",
      "blocks.3.norm.weight",
      "blocks.3.norm.bias",
      "blocks.3.reduction.bias",
      "blocks.4.norm2.weight",
      "blocks.4.norm2.bias",
      "blocks.4.mlp.fc1.bias",
      "blocks.4.mlp.ffn_ln.weight",
      "blocks.4.mlp.ffn_ln.bias",
      "blocks.4.mlp.fc2.bias",
      "blocks.5.norm2.weight",
      "blocks.5.norm2.bias",
      "blocks.5.mlp.fc1.bias",
      "blocks.5.mlp.ffn_ln.weight",
      "blocks.5.mlp.ffn_ln.bias",
      "blocks.5.mlp.fc2.bias",
      "blocks.6.norm2.weight",
      "blocks.6.norm2.bias",
      "blocks.6.mlp.fc1.bias",
      "blocks.6.mlp.ffn_ln.weight",
      "blocks.6.mlp.ffn_ln.bias",
      "blocks.6.mlp.fc2.bias",
      "blocks.7.norm.weight",
      "blocks.7.norm.bias",
      "blocks.7.reduction.bias",
      "blocks.8.gamma_1",
      "blocks.8.gamma_2",
      "blocks.8.norm1.weight",
      "blocks.8.norm1.bias",
      "blocks.8.attn.q_bias",
      "blocks.8.attn.v_bias",
      "blocks.8.attn.proj.bias",
      "blocks.8.norm2.weight",
      "blocks.8.norm2.bias",
      "blocks.8.mlp.w1.bias",
      "blocks.8.mlp.w2.bias",
      "blocks.8.mlp.ffn_ln.weight",
      "blocks.8.mlp.ffn_ln.bias",
      "blocks.8.mlp.w3.bias",
      "blocks.9.gamma_1",
      "blocks.9.gamma_2",
      "blocks.9.norm1.weight",
      "blocks.9.norm1.bias",
      "blocks.9.attn.q_bias",
      "blocks.9.attn.v_bias",
      "blocks.9.attn.proj.bias",
      "blocks.9.norm2.weight",
      "blocks.9.norm2.bias",
      "blocks.9.mlp.w1.bias",
      "blocks.9.mlp.w2.bias",
      "blocks.9.mlp.ffn_ln.weight",
      "blocks.9.mlp.ffn_ln.bias",
      "blocks.9.mlp.w3.bias",
      "blocks.10.gamma_1",
      "blocks.10.gamma_2",
      "blocks.10.norm1.weight",
      "blocks.10.norm1.bias",
      "blocks.10.attn.q_bias",
      "blocks.10.attn.v_bias",
      "blocks.10.attn.proj.bias",
      "blocks.10.norm2.weight",
      "blocks.10.norm2.bias",
      "blocks.10.mlp.w1.bias",
      "blocks.10.mlp.w2.bias",
      "blocks.10.mlp.ffn_ln.weight",
      "blocks.10.mlp.ffn_ln.bias",
      "blocks.10.mlp.w3.bias",
      "blocks.11.gamma_1",
      "blocks.11.gamma_2",
      "blocks.11.norm1.weight",
      "blocks.11.norm1.bias",
      "blocks.11.attn.q_bias",
      "blocks.11.attn.v_bias",
      "blocks.11.attn.proj.bias",
      "blocks.11.norm2.weight",
      "blocks.11.norm2.bias",
      "blocks.11.mlp.w1.bias",
      "blocks.11.mlp.w2.bias",
      "blocks.11.mlp.ffn_ln.weight",
      "blocks.11.mlp.ffn_ln.bias",
      "blocks.11.mlp.w3.bias",
      "blocks.12.gamma_1",
      "blocks.12.gamma_2",
      "blocks.12.norm1.weight",
      "blocks.12.norm1.bias",
      "blocks.12.attn.q_bias",
      "blocks.12.attn.v_bias",
      "blocks.12.attn.proj.bias",
      "blocks.12.norm2.weight",
      "blocks.12.norm2.bias",
      "blocks.12.mlp.w1.bias",
      "blocks.12.mlp.w2.bias",
      "blocks.12.mlp.ffn_ln.weight",
      "blocks.12.mlp.ffn_ln.bias",
      "blocks.12.mlp.w3.bias",
      "blocks.13.gamma_1",
      "blocks.13.gamma_2",
      "blocks.13.norm1.weight",
      "blocks.13.norm1.bias",
      "blocks.13.attn.q_bias",
      "blocks.13.attn.v_bias",
      "blocks.13.attn.proj.bias",
      "blocks.13.norm2.weight",
      "blocks.13.norm2.bias",
      "blocks.13.mlp.w1.bias",
      "blocks.13.mlp.w2.bias",
      "blocks.13.mlp.ffn_ln.weight",
      "blocks.13.mlp.ffn_ln.bias",
      "blocks.13.mlp.w3.bias",
      "blocks.14.gamma_1",
      "blocks.14.gamma_2",
      "blocks.14.norm1.weight",
      "blocks.14.norm1.bias",
      "blocks.14.attn.q_bias",
      "blocks.14.attn.v_bias",
      "blocks.14.attn.proj.bias",
      "blocks.14.norm2.weight",
      "blocks.14.norm2.bias",
      "blocks.14.mlp.w1.bias",
      "blocks.14.mlp.w2.bias",
      "blocks.14.mlp.ffn_ln.weight",
      "blocks.14.mlp.ffn_ln.bias",
      "blocks.14.mlp.w3.bias",
      "blocks.15.gamma_1",
      "blocks.15.gamma_2",
      "blocks.15.norm1.weight",
      "blocks.15.norm1.bias",
      "blocks.15.attn.q_bias",
      "blocks.15.attn.v_bias",
      "blocks.15.attn.proj.bias",
      "blocks.15.norm2.weight",
      "blocks.15.norm2.bias",
      "blocks.15.mlp.w1.bias",
      "blocks.15.mlp.w2.bias",
      "blocks.15.mlp.ffn_ln.weight",
      "blocks.15.mlp.ffn_ln.bias",
      "blocks.15.mlp.w3.bias",
      "blocks.16.gamma_1",
      "blocks.16.gamma_2",
      "blocks.16.norm1.weight",
      "blocks.16.norm1.bias",
      "blocks.16.attn.q_bias",
      "blocks.16.attn.v_bias",
      "blocks.16.attn.proj.bias",
      "blocks.16.norm2.weight",
      "blocks.16.norm2.bias",
      "blocks.16.mlp.w1.bias",
      "blocks.16.mlp.w2.bias",
      "blocks.16.mlp.ffn_ln.weight",
      "blocks.16.mlp.ffn_ln.bias",
      "blocks.16.mlp.w3.bias",
      "blocks.17.gamma_1",
      "blocks.17.gamma_2",
      "blocks.17.norm1.weight",
      "blocks.17.norm1.bias",
      "blocks.17.attn.q_bias",
      "blocks.17.attn.v_bias",
      "blocks.17.attn.proj.bias",
      "blocks.17.norm2.weight",
      "blocks.17.norm2.bias",
      "blocks.17.mlp.w1.bias",
      "blocks.17.mlp.w2.bias",
      "blocks.17.mlp.ffn_ln.weight",
      "blocks.17.mlp.ffn_ln.bias",
      "blocks.17.mlp.w3.bias",
      "blocks.18.gamma_1",
      "blocks.18.gamma_2",
      "blocks.18.norm1.weight",
      "blocks.18.norm1.bias",
      "blocks.18.attn.q_bias",
      "blocks.18.attn.v_bias",
      "blocks.18.attn.proj.bias",
      "blocks.18.norm2.weight",
      "blocks.18.norm2.bias",
      "blocks.18.mlp.w1.bias",
      "blocks.18.mlp.w2.bias",
      "blocks.18.mlp.ffn_ln.weight",
      "blocks.18.mlp.ffn_ln.bias",
      "blocks.18.mlp.w3.bias",
      "blocks.19.gamma_1",
      "blocks.19.gamma_2",
      "blocks.19.norm1.weight",
      "blocks.19.norm1.bias",
      "blocks.19.attn.q_bias",
      "blocks.19.attn.v_bias",
      "blocks.19.attn.proj.bias",
      "blocks.19.norm2.weight",
      "blocks.19.norm2.bias",
      "blocks.19.mlp.w1.bias",
      "blocks.19.mlp.w2.bias",
      "blocks.19.mlp.ffn_ln.weight",
      "blocks.19.mlp.ffn_ln.bias",
      "blocks.19.mlp.w3.bias",
      "blocks.20.gamma_1",
      "blocks.20.gamma_2",
      "blocks.20.norm1.weight",
      "blocks.20.norm1.bias",
      "blocks.20.attn.q_bias",
      "blocks.20.attn.v_bias",
      "blocks.20.attn.proj.bias",
      "blocks.20.norm2.weight",
      "blocks.20.norm2.bias",
      "blocks.20.mlp.w1.bias",
      "blocks.20.mlp.w2.bias",
      "blocks.20.mlp.ffn_ln.weight",
      "blocks.20.mlp.ffn_ln.bias",
      "blocks.20.mlp.w3.bias",
      "blocks.21.gamma_1",
      "blocks.21.gamma_2",
      "blocks.21.norm1.weight",
      "blocks.21.norm1.bias",
      "blocks.21.attn.q_bias",
      "blocks.21.attn.v_bias",
      "blocks.21.attn.proj.bias",
      "blocks.21.norm2.weight",
      "blocks.21.norm2.bias",
      "blocks.21.mlp.w1.bias",
      "blocks.21.mlp.w2.bias",
      "blocks.21.mlp.ffn_ln.weight",
      "blocks.21.mlp.ffn_ln.bias",
      "blocks.21.mlp.w3.bias",
      "blocks.22.gamma_1",
      "blocks.22.gamma_2",
      "blocks.22.norm1.weight",
      "blocks.22.norm1.bias",
      "blocks.22.attn.q_bias",
      "blocks.22.attn.v_bias",
      "blocks.22.attn.proj.bias",
      "blocks.22.norm2.weight",
      "blocks.22.norm2.bias",
      "blocks.22.mlp.w1.bias",
      "blocks.22.mlp.w2.bias",
      "blocks.22.mlp.ffn_ln.weight",
      "blocks.22.mlp.ffn_ln.bias",
      "blocks.22.mlp.w3.bias",
      "blocks.23.gamma_1",
      "blocks.23.gamma_2",
      "blocks.23.norm1.weight",
      "blocks.23.norm1.bias",
      "blocks.23.attn.q_bias",
      "blocks.23.attn.v_bias",
      "blocks.23.attn.proj.bias",
      "blocks.23.norm2.weight",
      "blocks.23.norm2.bias",
      "blocks.23.mlp.w1.bias",
      "blocks.23.mlp.w2.bias",
      "blocks.23.mlp.ffn_ln.weight",
      "blocks.23.mlp.ffn_ln.bias",
      "blocks.23.mlp.w3.bias",
      "blocks.24.gamma_1",
      "blocks.24.gamma_2",
      "blocks.24.norm1.weight",
      "blocks.24.norm1.bias",
      "blocks.24.attn.q_bias",
      "blocks.24.attn.v_bias",
      "blocks.24.attn.proj.bias",
      "blocks.24.norm2.weight",
      "blocks.24.norm2.bias",
      "blocks.24.mlp.w1.bias",
      "blocks.24.mlp.w2.bias",
      "blocks.24.mlp.ffn_ln.weight",
      "blocks.24.mlp.ffn_ln.bias",
      "blocks.24.mlp.w3.bias",
      "blocks.25.gamma_1",
      "blocks.25.gamma_2",
      "blocks.25.norm1.weight",
      "blocks.25.norm1.bias",
      "blocks.25.attn.q_bias",
      "blocks.25.attn.v_bias",
      "blocks.25.attn.proj.bias",
      "blocks.25.norm2.weight",
      "blocks.25.norm2.bias",
      "blocks.25.mlp.w1.bias",
      "blocks.25.mlp.w2.bias",
      "blocks.25.mlp.ffn_ln.weight",
      "blocks.25.mlp.ffn_ln.bias",
      "blocks.25.mlp.w3.bias",
      "blocks.26.gamma_1",
      "blocks.26.gamma_2",
      "blocks.26.norm1.weight",
      "blocks.26.norm1.bias",
      "blocks.26.attn.q_bias",
      "blocks.26.attn.v_bias",
      "blocks.26.attn.proj.bias",
      "blocks.26.norm2.weight",
      "blocks.26.norm2.bias",
      "blocks.26.mlp.w1.bias",
      "blocks.26.mlp.w2.bias",
      "blocks.26.mlp.ffn_ln.weight",
      "blocks.26.mlp.ffn_ln.bias",
      "blocks.26.mlp.w3.bias",
      "blocks.27.gamma_1",
      "blocks.27.gamma_2",
      "blocks.27.norm1.weight",
      "blocks.27.norm1.bias",
      "blocks.27.attn.q_bias",
      "blocks.27.attn.v_bias",
      "blocks.27.attn.proj.bias",
      "blocks.27.norm2.weight",
      "blocks.27.norm2.bias",
      "blocks.27.mlp.w1.bias",
      "blocks.27.mlp.w2.bias",
      "blocks.27.mlp.ffn_ln.weight",
      "blocks.27.mlp.ffn_ln.bias",
      "blocks.27.mlp.w3.bias",
      "blocks.28.gamma_1",
      "blocks.28.gamma_2",
      "blocks.28.norm1.weight",
      "blocks.28.norm1.bias",
      "blocks.28.attn.q_bias",
      "blocks.28.attn.v_bias",
      "blocks.28.attn.proj.bias",
      "blocks.28.norm2.weight",
      "blocks.28.norm2.bias",
      "blocks.28.mlp.w1.bias",
      "blocks.28.mlp.w2.bias",
      "blocks.28.mlp.ffn_ln.weight",
      "blocks.28.mlp.ffn_ln.bias",
      "blocks.28.mlp.w3.bias",
      "blocks.29.gamma_1",
      "blocks.29.gamma_2",
      "blocks.29.norm1.weight",
      "blocks.29.norm1.bias",
      "blocks.29.attn.q_bias",
      "blocks.29.attn.v_bias",
      "blocks.29.attn.proj.bias",
      "blocks.29.norm2.weight",
      "blocks.29.norm2.bias",
      "blocks.29.mlp.w1.bias",
      "blocks.29.mlp.w2.bias",
      "blocks.29.mlp.ffn_ln.weight",
      "blocks.29.mlp.ffn_ln.bias",
      "blocks.29.mlp.w3.bias",
      "blocks.30.gamma_1",
      "blocks.30.gamma_2",
      "blocks.30.norm1.weight",
      "blocks.30.norm1.bias",
      "blocks.30.attn.q_bias",
      "blocks.30.attn.v_bias",
      "blocks.30.attn.proj.bias",
      "blocks.30.norm2.weight",
      "blocks.30.norm2.bias",
      "blocks.30.mlp.w1.bias",
      "blocks.30.mlp.w2.bias",
      "blocks.30.mlp.ffn_ln.weight",
      "blocks.30.mlp.ffn_ln.bias",
      "blocks.30.mlp.w3.bias",
      "blocks.31.gamma_1",
      "blocks.31.gamma_2",
      "blocks.31.norm1.weight",
      "blocks.31.norm1.bias",
      "blocks.31.attn.q_bias",
      "blocks.31.attn.v_bias",
      "blocks.31.attn.proj.bias",
      "blocks.31.norm2.weight",
      "blocks.31.norm2.bias",
      "blocks.31.mlp.w1.bias",
      "blocks.31.mlp.w2.bias",
      "blocks.31.mlp.ffn_ln.weight",
      "blocks.31.mlp.ffn_ln.bias",
      "blocks.31.mlp.w3.bias",
      "align_dim_16tofpn.bias",
      "fpn_modules.0.norm2.weight",
      "fpn_modules.0.norm2.bias",
      "fpn_modules.0.mlp.fc1.bias",
      "fpn_modules.0.mlp.fc2.bias",
      "fpn_modules.1.norm2.weight",
      "fpn_modules.1.norm2.bias",
      "fpn_modules.1.mlp.fc1.bias",
      "fpn_modules.1.mlp.fc2.bias",
      "fpn_modules.2.norm2.weight",
      "fpn_modules.2.norm2.bias",
      "fpn_modules.2.mlp.fc1.bias",
      "fpn_modules.2.mlp.fc2.bias",
      "align_dim_16to8.bias",
      "split_16to8.norm.weight",
      "split_16to8.norm.bias",
      "split_16to8.reduction.bias",
      "block_16to8.0.norm2.weight",
      "block_16to8.0.norm2.bias",
      "block_16to8.0.mlp.fc1.bias",
      "block_16to8.0.mlp.fc2.bias",
      "align_dim_8to4.bias",
      "split_8to4.norm.weight",
      "split_8to4.norm.bias",
      "split_8to4.reduction.bias",
      "block_8to4.0.norm2.weight",
      "block_8to4.0.norm2.bias",
      "block_8to4.0.mlp.fc1.bias",
      "block_8to4.0.mlp.fc2.bias",
      "decoder_embed.0.0.weight",
      "decoder_embed.0.0.bias",
      "decoder_embed.0.1.bias",
      "decoder_embed.1.0.weight",
      "decoder_embed.1.0.bias",
      "decoder_embed.1.1.bias",
      "decoder_embed.2.0.weight",
      "decoder_embed.2.0.bias",
      "decoder_embed.2.1.bias",
      "norm.weight",
      "norm.bias",
      "lm_head.bias"
    ],
    "lr_scale": 1.0
  }
}
[2024-07-24 02:13:16,003] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.4, git-hash=unknown, git-branch=unknown
[2024-07-24 02:13:16,003] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-24 02:13:16,013] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.4, git-hash=unknown, git-branch=unknown
[2024-07-24 02:13:16,013] [INFO] [comm.py:637:init_distributed] cdb=None
Loading dataset shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 257/257 [00:00<00:00, 19489.70it/s]
[2024-07-24 02:13:16,133] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.4, git-hash=unknown, git-branch=unknown
[2024-07-24 02:13:16,133] [INFO] [comm.py:637:init_distributed] cdb=None
[2024-07-24 02:13:16,307] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: True
Using /home/xiaofeng.wu/.cache/torch_extensions/py310_cu121 as PyTorch extensions root...
Detected CUDA files, patching ldflags
Emitting ninja build file /home/xiaofeng.wu/.cache/torch_extensions/py310_cu121/fused_adam/build.ninja...
/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/utils/cpp_extension.py:1967: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.
If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'].
  warnings.warn(
Building extension module fused_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
ninja: no work to do.
Loading extension module fused_adam...
Time to load fused_adam op: 0.049057722091674805 seconds
[2024-07-24 02:13:16,357] [INFO] [logging.py:96:log_dist] [Rank 0] Using DeepSpeed Optimizer param name adam as basic optimizer
[2024-07-24 02:13:16,357] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2024-07-24 02:13:16,389] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = FusedAdam
[2024-07-24 02:13:16,389] [INFO] [utils.py:56:is_zero_supported_optimizer] Checking ZeRO support for optimizer=FusedAdam type=<class 'deepspeed.ops.adam.fused_adam.FusedAdam'>
[2024-07-24 02:13:16,389] [INFO] [logging.py:96:log_dist] [Rank 0] Creating torch.float16 ZeRO stage 1 optimizer
[2024-07-24 02:13:16,389] [INFO] [stage_1_and_2.py:148:__init__] Reduce bucket size 500000000
[2024-07-24 02:13:16,389] [INFO] [stage_1_and_2.py:149:__init__] Allgather bucket size 500,000,000
[2024-07-24 02:13:16,389] [INFO] [stage_1_and_2.py:150:__init__] CPU Offload: False
[2024-07-24 02:13:16,389] [INFO] [stage_1_and_2.py:151:__init__] Round robin gradient partitioning: False
[2024-07-24 02:13:16,632] [INFO] [utils.py:781:see_memory_usage] Before initializing optimizer states
[2024-07-24 02:13:16,632] [INFO] [utils.py:782:see_memory_usage] MA 0.81 GB         Max_MA 0.81 GB         CA 0.88 GB         Max_CA 1 GB
[2024-07-24 02:13:16,632] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 17.18 GB, percent = 6.8%
[2024-07-24 02:13:16,738] [INFO] [utils.py:781:see_memory_usage] After initializing optimizer states
[2024-07-24 02:13:16,738] [INFO] [utils.py:782:see_memory_usage] MA 0.81 GB         Max_MA 0.9 GB         CA 0.96 GB         Max_CA 1 GB
[2024-07-24 02:13:16,739] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 17.19 GB, percent = 6.8%
[2024-07-24 02:13:16,739] [INFO] [stage_1_and_2.py:543:__init__] optimizer state initialized
[2024-07-24 02:13:16,843] [INFO] [utils.py:781:see_memory_usage] After initializing ZeRO optimizer
[2024-07-24 02:13:16,843] [INFO] [utils.py:782:see_memory_usage] MA 0.81 GB         Max_MA 0.81 GB         CA 0.96 GB         Max_CA 1 GB
[2024-07-24 02:13:16,844] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 17.2 GB, percent = 6.8%
[2024-07-24 02:13:16,845] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer
[2024-07-24 02:13:16,845] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2024-07-24 02:13:16,845] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2024-07-24 02:13:16,845] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[0.0015, 0.0015], mom=[[0.9, 0.98], [0.9, 0.98]]
[2024-07-24 02:13:16,846] [INFO] [config.py:997:print] DeepSpeedEngine configuration:
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   amp_enabled .................. False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   amp_params ................... {'opt_level': 'O2'}
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   autotuning_config ............ {
    "enabled": false,
    "start_step": null,
    "end_step": null,
    "metric_path": null,
    "arg_mappings": null,
    "metric": "throughput",
    "model_info": null,
    "results_dir": "autotuning_results",
    "exps_dir": "autotuning_exps",
    "overwrite": true,
    "fast": true,
    "start_profile_step": 3,
    "end_profile_step": 5,
    "tuner_type": "gridsearch",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "model_info_path": null,
    "mp_size": 1,
    "max_train_batch_size": null,
    "min_train_batch_size": 1,
    "max_train_micro_batch_size_per_gpu": 1.024000e+03,
    "min_train_micro_batch_size_per_gpu": 1,
    "num_tuning_micro_batch_sizes": 3
}
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   bfloat16_enabled ............. False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   bfloat16_immediate_grad_update  False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   checkpoint_parallel_write_pipeline  False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   checkpoint_tag_validation_enabled  True
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   checkpoint_tag_validation_fail  False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7f59cc0a1d20>
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   communication_data_type ...... None
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   curriculum_enabled_legacy .... False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   curriculum_params_legacy ..... False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   data_efficiency_enabled ...... False
[2024-07-24 02:13:16,846] [INFO] [config.py:1001:print]   dataloader_drop_last ......... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   disable_allgather ............ False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   dump_state ................... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   dynamic_loss_scale_args ...... {'init_scale': 65536, 'scale_window': 500, 'delayed_shift': 2, 'consecutive_hysteresis': False, 'min_scale': 1}
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_enabled ........... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_gas_boundary_resolution  1
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_layer_num ......... 0
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_max_iter .......... 100
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_stability ......... 1e-06
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_tol ............... 0.01
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   eigenvalue_verbose ........... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   elasticity_enabled ........... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   flops_profiler_config ........ {
    "enabled": true,
    "recompute_fwd_factor": 0.0,
    "profile_step": -1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
}
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   fp16_auto_cast ............... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   fp16_enabled ................. True
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   fp16_master_weights_and_gradients  False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   global_rank .................. 0
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   grad_accum_dtype ............. None
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   gradient_accumulation_steps .. 1
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   gradient_clipping ............ 3
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   gradient_predivide_factor .... 1.0
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   graph_harvesting ............. False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   initial_dynamic_scale ........ 65536
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   load_universal_checkpoint .... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   loss_scale ................... 0
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   memory_breakdown ............. False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   mics_hierarchial_params_gather  False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   mics_shard_size .............. -1
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') comet=CometConfig(enabled=False, samples_log_interval=100, project=None, workspace=None, api_key=None, experiment_name=None, experiment_key=None, online=None, mode=None) wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') enabled=False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   optimizer_legacy_fusion ...... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   optimizer_name ............... adam
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   optimizer_params ............. {'lr': 0.0015, 'weight_decay': 0.05, 'bias_correction': True, 'betas': [0.9, 0.98], 'eps': 1e-08}
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   pld_enabled .................. False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   pld_params ................... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   prescale_gradients ........... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   scheduler_name ............... None
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   scheduler_params ............. None
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   seq_parallel_communication_data_type  torch.float32
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   sparse_attention ............. None
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   sparse_gradients_enabled ..... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   steps_per_print .............. 1000
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   timers_config ................ enabled=True synchronized=True
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   train_batch_size ............. 128
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   train_micro_batch_size_per_gpu  32
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   use_data_before_expert_parallel_  False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   use_node_local_storage ....... False
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   wall_clock_breakdown ......... True
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   weight_quantization_config ... None
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   world_size ................... 4
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   zero_allow_untested_optimizer  True
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   zero_config .................. stage=1 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500000000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500,000,000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1,000,000,000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50,000,000 param_persistence_threshold=100,000 model_persistence_threshold=sys.maxsize max_live_parameters=1,000,000,000 max_reuse_distance=1,000,000,000 gather_16bit_weights_on_model_save=False use_all_reduce_for_fetch_params=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   zero_enabled ................. True
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   zero_force_ds_cpu_optimizer .. True
[2024-07-24 02:13:16,847] [INFO] [config.py:1001:print]   zero_optimization_stage ...... 1
[2024-07-24 02:13:16,847] [INFO] [config.py:987:print_user_config]   json = {
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 32,
    "steps_per_print": 1000,
    "optimizer": {
        "type": "Adam",
        "adam_w_mode": true,
        "params": {
            "lr": 0.0015,
            "weight_decay": 0.05,
            "bias_correction": true,
            "betas": [0.9, 0.98],
            "eps": 1e-08
        }
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 500,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": false
    },
    "amp": {
        "enabled": false,
        "opt_level": "O2"
    },
    "flops_profiler": {
        "enabled": true,
        "profile_step": -1,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": true
    },
    "zero_allow_untested_optimizer": true,
    "gradient_clipping": 3,
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5.000000e+08
    }
}
model.gradient_accumulation_steps() = 1
Model = DeepSpeedEngine(
  (module): iTPNForMIM(
    (patch_embed): ConvPatchEmbed(
      (proj): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (blocks): ModuleList(
      (0): ConvMlpBlock(
        (drop_path): Identity()
        (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(128, 384, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (ffn_ln): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
          (fc2): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (1): ConvMlpBlock(
        (drop_path): DropPath(p=0.003448275849223137)
        (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(128, 384, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (ffn_ln): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
          (fc2): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (2): ConvMlpBlock(
        (drop_path): DropPath(p=0.006896551698446274)
        (norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(128, 384, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (ffn_ln): LayerNorm((384,), eps=1e-06, elementwise_affine=True)
          (fc2): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (3): ConvPatchMerge(
        (norm): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
        (reduction): Conv2d(128, 256, kernel_size=(2, 2), stride=(2, 2))
      )
      (4): ConvMlpBlock(
        (drop_path): DropPath(p=0.01034482754766941)
        (norm2): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (ffn_ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (fc2): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (5): ConvMlpBlock(
        (drop_path): DropPath(p=0.013793103396892548)
        (norm2): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (ffn_ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (fc2): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (6): ConvMlpBlock(
        (drop_path): DropPath(p=0.017241379246115685)
        (norm2): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (ffn_ln): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
          (fc2): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (7): ConvPatchMerge(
        (norm): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (reduction): Conv2d(256, 512, kernel_size=(2, 2), stride=(2, 2))
      )
      (8): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.02068965509533882)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (9): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.02413793094456196)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (10): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.027586206793785095)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (11): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.031034482643008232)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (12): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.03448275849223137)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (13): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.03793103247880936)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (14): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.04137931019067764)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (15): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.04482758790254593)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (16): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.04827586188912392)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (17): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.0517241396009922)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (18): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.05517241358757019)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (19): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.05862069129943848)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (20): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.06206896901130676)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (21): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.06551724672317505)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (22): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.06896551698446274)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (23): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.07241379469633102)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (24): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.07586207240819931)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (25): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.079310342669487)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (26): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.08275862038135529)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (27): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.08620689809322357)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (28): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.08965517580509186)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (29): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.09310345351696014)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (30): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.09655172377824783)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (31): Block(
        (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (attn): Attention(
          (q_proj): Linear(in_features=512, out_features=512, bias=False)
          (k_proj): Linear(in_features=512, out_features=512, bias=False)
          (v_proj): Linear(in_features=512, out_features=512, bias=False)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=512, out_features=512, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
        )
        (drop_path): DropPath(p=0.10000000149011612)
        (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (mlp): SwiGLU(
          (w1): Linear(in_features=512, out_features=1536, bias=True)
          (w2): Linear(in_features=512, out_features=1536, bias=True)
          (act): SiLU()
          (ffn_ln): LayerNorm((1536,), eps=1e-06, elementwise_affine=True)
          (w3): Linear(in_features=1536, out_features=512, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (align_dim_16tofpn): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
    (fpn_modules): ModuleList(
      (0-2): 3 x ConvMlpBlock(
        (drop_path): Identity()
        (norm2): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(256, 682, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (fc2): Conv2d(682, 256, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (align_dim_16to8): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (split_16to8): ConvPatchSplit(
      (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
      (reduction): ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
    )
    (block_16to8): Sequential(
      (0): ConvMlpBlock(
        (drop_path): Identity()
        (norm2): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(256, 682, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (fc2): Conv2d(682, 256, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (align_dim_8to4): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
    (split_8to4): ConvPatchSplit(
      (norm): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
      (reduction): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
    )
    (block_8to4): Sequential(
      (0): ConvMlpBlock(
        (drop_path): Identity()
        (norm2): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (mlp): ConvMlp(
          (fc1): Conv2d(256, 682, kernel_size=(1, 1), stride=(1, 1))
          (act): GELU(approximate='none')
          (fc2): Conv2d(682, 256, kernel_size=(1, 1), stride=(1, 1))
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (decoder_embed): ModuleList(
      (0): ModuleList(
        (0): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (1): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ModuleList(
        (0): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (1): Conv2d(256, 512, kernel_size=(2, 2), stride=(2, 2))
      )
      (2): ModuleList(
        (0): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
        (1): Conv2d(256, 512, kernel_size=(4, 4), stride=(4, 4))
      )
    )
    (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    (lm_head): Linear(in_features=512, out_features=768, bias=True)
  )
)
number of params: 90235602
Optimizer = <deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer object at 0x7f59cc0a03a0>
Use step level LR & WD scheduler!
Set warmup steps = 100090
args.weight_decay: 0.05
args.weight_decay_end: 0.05
args.epochs: 300
num_training_steps_per_epoch: 10009
Set warmup steps = 0
ls: cannot access '/cache/output': No such file or directory
ls: cannot access '/cache/output': No such file or directory
ls: cannot access '/cache/output': No such file or directory
wd_schedule_values: [0.05 0.05 0.05 ... 0.05 0.05 0.05]
Max WD = 0.0500000, Min WD = 0.0500000
start auto load model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
start auto load model ccccccccccccccc >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
output_dir : /tmp/output
ls: cannot access '/cache/output': No such file or directory
all_checkpoints: []
Check parameter scale !
module.mask_token: torch.Size([1, 1, 2048]) -0.00029969, 0.00917816, 0.03997803, require_grad = True
module.pos_embed: torch.Size([1, 196, 512]) -0.00002843, 0.00919342, 0.04000854, require_grad = True
module.patch_embed.proj.weight: torch.Size([128, 3, 4, 4]) 0.00011945, 0.09277344, 0.36914062, require_grad = True
module.patch_embed.proj.bias: torch.Size([128]) -0.00923920, 0.06793213, 0.28369141, require_grad = True
module.blocks.0.norm2.weight: torch.Size([128]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.0.norm2.bias: torch.Size([128]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.0.mlp.fc1.weight: torch.Size([384, 128, 1, 1]) 0.00067091, 0.08697510, 0.34790039, require_grad = True
module.blocks.0.mlp.fc1.bias: torch.Size([384]) 0.00374985, 0.08453369, 0.34570312, require_grad = True
module.blocks.0.mlp.ffn_ln.weight: torch.Size([384]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.0.mlp.ffn_ln.bias: torch.Size([384]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.0.mlp.fc2.weight: torch.Size([128, 384, 1, 1]) 0.00024939, 0.05020142, 0.20080566, require_grad = True
module.blocks.0.mlp.fc2.bias: torch.Size([128]) 0.00784302, 0.04937744, 0.19921875, require_grad = True
module.blocks.1.norm2.weight: torch.Size([128]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.1.norm2.bias: torch.Size([128]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.1.mlp.fc1.weight: torch.Size([384, 128, 1, 1]) -0.00068665, 0.08715820, 0.34790039, require_grad = True
module.blocks.1.mlp.fc1.bias: torch.Size([384]) 0.00808716, 0.08251953, 0.34570312, require_grad = True
module.blocks.1.mlp.ffn_ln.weight: torch.Size([384]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.1.mlp.ffn_ln.bias: torch.Size([384]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.1.mlp.fc2.weight: torch.Size([128, 384, 1, 1]) -0.00031567, 0.05020142, 0.20080566, require_grad = True
module.blocks.1.mlp.fc2.bias: torch.Size([128]) 0.00085783, 0.05038452, 0.19799805, require_grad = True
module.blocks.2.norm2.weight: torch.Size([128]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.2.norm2.bias: torch.Size([128]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.2.mlp.fc1.weight: torch.Size([384, 128, 1, 1]) 0.00014436, 0.08758545, 0.34790039, require_grad = True
module.blocks.2.mlp.fc1.bias: torch.Size([384]) 0.00238419, 0.08831787, 0.34667969, require_grad = True
module.blocks.2.mlp.ffn_ln.weight: torch.Size([384]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.2.mlp.ffn_ln.bias: torch.Size([384]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.2.mlp.fc2.weight: torch.Size([128, 384, 1, 1]) -0.00035453, 0.05004883, 0.20080566, require_grad = True
module.blocks.2.mlp.fc2.bias: torch.Size([128]) -0.00372887, 0.04974365, 0.19958496, require_grad = True
module.blocks.3.norm.weight: torch.Size([128]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.3.norm.bias: torch.Size([128]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.3.reduction.weight: torch.Size([256, 128, 2, 2]) -0.00013077, 0.02204895, 0.08837891, require_grad = True
module.blocks.3.reduction.bias: torch.Size([256]) 0.00016832, 0.02262878, 0.08776855, require_grad = True
module.blocks.4.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.4.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.4.mlp.fc1.weight: torch.Size([768, 256, 1, 1]) 0.00001979, 0.06149292, 0.24597168, require_grad = True
module.blocks.4.mlp.fc1.bias: torch.Size([768]) 0.00177002, 0.06155396, 0.24560547, require_grad = True
module.blocks.4.mlp.ffn_ln.weight: torch.Size([768]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.4.mlp.ffn_ln.bias: torch.Size([768]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.4.mlp.fc2.weight: torch.Size([256, 768, 1, 1]) 0.00001955, 0.03555298, 0.14196777, require_grad = True
module.blocks.4.mlp.fc2.bias: torch.Size([256]) -0.00300598, 0.03347778, 0.14135742, require_grad = True
module.blocks.5.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.5.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.5.mlp.fc1.weight: torch.Size([768, 256, 1, 1]) -0.00007683, 0.06143188, 0.24597168, require_grad = True
module.blocks.5.mlp.fc1.bias: torch.Size([768]) -0.00269508, 0.06317139, 0.24560547, require_grad = True
module.blocks.5.mlp.ffn_ln.weight: torch.Size([768]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.5.mlp.ffn_ln.bias: torch.Size([768]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.5.mlp.fc2.weight: torch.Size([256, 768, 1, 1]) -0.00009722, 0.03543091, 0.14196777, require_grad = True
module.blocks.5.mlp.fc2.bias: torch.Size([256]) 0.00460815, 0.03616333, 0.14135742, require_grad = True
module.blocks.6.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.6.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.6.mlp.fc1.weight: torch.Size([768, 256, 1, 1]) 0.00016439, 0.06149292, 0.24597168, require_grad = True
module.blocks.6.mlp.fc1.bias: torch.Size([768]) 0.00119495, 0.06060791, 0.24560547, require_grad = True
module.blocks.6.mlp.ffn_ln.weight: torch.Size([768]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.6.mlp.ffn_ln.bias: torch.Size([768]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.6.mlp.fc2.weight: torch.Size([256, 768, 1, 1]) 0.00001299, 0.03546143, 0.14196777, require_grad = True
module.blocks.6.mlp.fc2.bias: torch.Size([256]) 0.00006765, 0.03619385, 0.14147949, require_grad = True
module.blocks.7.norm.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.7.norm.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.7.reduction.weight: torch.Size([512, 256, 2, 2]) -0.00002205, 0.01562500, 0.06250000, require_grad = True
module.blocks.7.reduction.bias: torch.Size([512]) -0.00093555, 0.01525879, 0.06231689, require_grad = True
module.blocks.8.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.8.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.8.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.8.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.attn.q_proj.weight: torch.Size([512, 512]) -0.00003618, 0.03527832, 0.40454102, require_grad = True
module.blocks.8.attn.k_proj.weight: torch.Size([512, 512]) -0.00006944, 0.03530884, 0.44555664, require_grad = True
module.blocks.8.attn.v_proj.weight: torch.Size([512, 512]) -0.00000775, 0.06945801, 0.80078125, require_grad = True
module.blocks.8.attn.proj.weight: torch.Size([512, 512]) 0.00016046, 0.06945801, 0.76904297, require_grad = True
module.blocks.8.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.8.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.mlp.w1.weight: torch.Size([1536, 512]) 0.00001156, 0.04907227, 0.58398438, require_grad = True
module.blocks.8.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.mlp.w2.weight: torch.Size([1536, 512]) -0.00000072, 0.04904175, 0.56201172, require_grad = True
module.blocks.8.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.8.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.8.mlp.w3.weight: torch.Size([512, 1536]) 0.00006139, 0.04910278, 0.59570312, require_grad = True
module.blocks.8.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.9.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.9.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.9.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.attn.q_proj.weight: torch.Size([512, 512]) 0.00005811, 0.03521729, 0.39794922, require_grad = True
module.blocks.9.attn.k_proj.weight: torch.Size([512, 512]) -0.00000924, 0.03521729, 0.40820312, require_grad = True
module.blocks.9.attn.v_proj.weight: torch.Size([512, 512]) 0.00027251, 0.06945801, 0.84814453, require_grad = True
module.blocks.9.attn.proj.weight: torch.Size([512, 512]) -0.00013793, 0.06933594, 0.81347656, require_grad = True
module.blocks.9.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.9.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.mlp.w1.weight: torch.Size([1536, 512]) -0.00004441, 0.04904175, 0.58203125, require_grad = True
module.blocks.9.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.mlp.w2.weight: torch.Size([1536, 512]) 0.00007558, 0.04901123, 0.56152344, require_grad = True
module.blocks.9.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.9.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.9.mlp.w3.weight: torch.Size([512, 1536]) -0.00009435, 0.04913330, 0.55078125, require_grad = True
module.blocks.9.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.10.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.10.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.10.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.attn.q_proj.weight: torch.Size([512, 512]) 0.00001538, 0.03527832, 0.40234375, require_grad = True
module.blocks.10.attn.k_proj.weight: torch.Size([512, 512]) 0.00008202, 0.03530884, 0.39184570, require_grad = True
module.blocks.10.attn.v_proj.weight: torch.Size([512, 512]) 0.00022948, 0.06933594, 0.79833984, require_grad = True
module.blocks.10.attn.proj.weight: torch.Size([512, 512]) -0.00002277, 0.06933594, 0.77343750, require_grad = True
module.blocks.10.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.10.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.mlp.w1.weight: torch.Size([1536, 512]) -0.00009203, 0.04904175, 0.58496094, require_grad = True
module.blocks.10.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.mlp.w2.weight: torch.Size([1536, 512]) 0.00009000, 0.04910278, 0.57812500, require_grad = True
module.blocks.10.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.10.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.10.mlp.w3.weight: torch.Size([512, 1536]) 0.00002241, 0.04898071, 0.59277344, require_grad = True
module.blocks.10.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.11.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.11.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.11.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.attn.q_proj.weight: torch.Size([512, 512]) -0.00000173, 0.03527832, 0.41040039, require_grad = True
module.blocks.11.attn.k_proj.weight: torch.Size([512, 512]) 0.00005209, 0.03530884, 0.39624023, require_grad = True
module.blocks.11.attn.v_proj.weight: torch.Size([512, 512]) 0.00003296, 0.06933594, 0.80419922, require_grad = True
module.blocks.11.attn.proj.weight: torch.Size([512, 512]) 0.00040865, 0.06945801, 0.78076172, require_grad = True
module.blocks.11.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.11.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.mlp.w1.weight: torch.Size([1536, 512]) 0.00002187, 0.04910278, 0.59521484, require_grad = True
module.blocks.11.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.mlp.w2.weight: torch.Size([1536, 512]) 0.00013196, 0.04901123, 0.57910156, require_grad = True
module.blocks.11.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.11.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.11.mlp.w3.weight: torch.Size([512, 1536]) 0.00001532, 0.04910278, 0.61328125, require_grad = True
module.blocks.11.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.12.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.12.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.12.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.attn.q_proj.weight: torch.Size([512, 512]) 0.00007367, 0.03518677, 0.40429688, require_grad = True
module.blocks.12.attn.k_proj.weight: torch.Size([512, 512]) -0.00005180, 0.03533936, 0.40380859, require_grad = True
module.blocks.12.attn.v_proj.weight: torch.Size([512, 512]) 0.00008422, 0.06945801, 0.81347656, require_grad = True
module.blocks.12.attn.proj.weight: torch.Size([512, 512]) -0.00010496, 0.06939697, 0.83642578, require_grad = True
module.blocks.12.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.12.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.mlp.w1.weight: torch.Size([1536, 512]) 0.00000846, 0.04910278, 0.59570312, require_grad = True
module.blocks.12.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.mlp.w2.weight: torch.Size([1536, 512]) 0.00008577, 0.04895020, 0.64013672, require_grad = True
module.blocks.12.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.12.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.12.mlp.w3.weight: torch.Size([512, 1536]) 0.00000042, 0.04901123, 0.58789062, require_grad = True
module.blocks.12.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.13.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.13.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.13.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.attn.q_proj.weight: torch.Size([512, 512]) 0.00003433, 0.03521729, 0.39990234, require_grad = True
module.blocks.13.attn.k_proj.weight: torch.Size([512, 512]) 0.00004441, 0.03524780, 0.41308594, require_grad = True
module.blocks.13.attn.v_proj.weight: torch.Size([512, 512]) 0.00004929, 0.06939697, 0.83789062, require_grad = True
module.blocks.13.attn.proj.weight: torch.Size([512, 512]) -0.00017023, 0.06939697, 0.79736328, require_grad = True
module.blocks.13.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.13.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.mlp.w1.weight: torch.Size([1536, 512]) 0.00004959, 0.04907227, 0.61718750, require_grad = True
module.blocks.13.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.mlp.w2.weight: torch.Size([1536, 512]) -0.00000870, 0.04901123, 0.56738281, require_grad = True
module.blocks.13.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.13.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.13.mlp.w3.weight: torch.Size([512, 1536]) -0.00000614, 0.04910278, 0.60156250, require_grad = True
module.blocks.13.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.14.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.14.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.14.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.attn.q_proj.weight: torch.Size([512, 512]) 0.00009358, 0.03530884, 0.41479492, require_grad = True
module.blocks.14.attn.k_proj.weight: torch.Size([512, 512]) 0.00001878, 0.03533936, 0.39501953, require_grad = True
module.blocks.14.attn.v_proj.weight: torch.Size([512, 512]) 0.00011384, 0.06933594, 0.78662109, require_grad = True
module.blocks.14.attn.proj.weight: torch.Size([512, 512]) 0.00004733, 0.06939697, 0.80566406, require_grad = True
module.blocks.14.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.14.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.mlp.w1.weight: torch.Size([1536, 512]) -0.00002921, 0.04904175, 0.62304688, require_grad = True
module.blocks.14.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.mlp.w2.weight: torch.Size([1536, 512]) -0.00006199, 0.04910278, 0.58593750, require_grad = True
module.blocks.14.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.14.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.14.mlp.w3.weight: torch.Size([512, 1536]) 0.00004858, 0.04904175, 0.54785156, require_grad = True
module.blocks.14.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.15.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.15.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.15.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.attn.q_proj.weight: torch.Size([512, 512]) -0.00002319, 0.03524780, 0.39160156, require_grad = True
module.blocks.15.attn.k_proj.weight: torch.Size([512, 512]) -0.00005877, 0.03518677, 0.41455078, require_grad = True
module.blocks.15.attn.v_proj.weight: torch.Size([512, 512]) -0.00006050, 0.06939697, 0.76806641, require_grad = True
module.blocks.15.attn.proj.weight: torch.Size([512, 512]) -0.00006092, 0.06945801, 0.73291016, require_grad = True
module.blocks.15.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.15.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.mlp.w1.weight: torch.Size([1536, 512]) -0.00002116, 0.04901123, 0.61035156, require_grad = True
module.blocks.15.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.mlp.w2.weight: torch.Size([1536, 512]) 0.00007415, 0.04913330, 0.63476562, require_grad = True
module.blocks.15.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.15.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.15.mlp.w3.weight: torch.Size([512, 1536]) 0.00001246, 0.04913330, 0.58593750, require_grad = True
module.blocks.15.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.16.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.16.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.16.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.attn.q_proj.weight: torch.Size([512, 512]) 0.00001448, 0.03524780, 0.39843750, require_grad = True
module.blocks.16.attn.k_proj.weight: torch.Size([512, 512]) 0.00000125, 0.03521729, 0.40429688, require_grad = True
module.blocks.16.attn.v_proj.weight: torch.Size([512, 512]) 0.00003505, 0.06939697, 0.79150391, require_grad = True
module.blocks.16.attn.proj.weight: torch.Size([512, 512]) -0.00010079, 0.06939697, 0.75537109, require_grad = True
module.blocks.16.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.16.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.mlp.w1.weight: torch.Size([1536, 512]) 0.00001496, 0.04895020, 0.56494141, require_grad = True
module.blocks.16.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.mlp.w2.weight: torch.Size([1536, 512]) 0.00003040, 0.04907227, 0.60449219, require_grad = True
module.blocks.16.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.16.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.16.mlp.w3.weight: torch.Size([512, 1536]) -0.00008357, 0.04901123, 0.60839844, require_grad = True
module.blocks.16.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.17.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.17.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.17.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.attn.q_proj.weight: torch.Size([512, 512]) -0.00005931, 0.03518677, 0.41894531, require_grad = True
module.blocks.17.attn.k_proj.weight: torch.Size([512, 512]) 0.00004041, 0.03527832, 0.40771484, require_grad = True
module.blocks.17.attn.v_proj.weight: torch.Size([512, 512]) -0.00032210, 0.06945801, 0.79199219, require_grad = True
module.blocks.17.attn.proj.weight: torch.Size([512, 512]) 0.00019491, 0.06927490, 0.80175781, require_grad = True
module.blocks.17.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.17.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.mlp.w1.weight: torch.Size([1536, 512]) -0.00010288, 0.04907227, 0.56640625, require_grad = True
module.blocks.17.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.mlp.w2.weight: torch.Size([1536, 512]) 0.00007439, 0.04904175, 0.57714844, require_grad = True
module.blocks.17.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.17.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.17.mlp.w3.weight: torch.Size([512, 1536]) -0.00011861, 0.04904175, 0.57128906, require_grad = True
module.blocks.17.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.18.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.18.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.18.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.attn.q_proj.weight: torch.Size([512, 512]) 0.00012708, 0.03524780, 0.40161133, require_grad = True
module.blocks.18.attn.k_proj.weight: torch.Size([512, 512]) 0.00003481, 0.03527832, 0.38330078, require_grad = True
module.blocks.18.attn.v_proj.weight: torch.Size([512, 512]) -0.00013161, 0.06921387, 0.79589844, require_grad = True
module.blocks.18.attn.proj.weight: torch.Size([512, 512]) 0.00001389, 0.06939697, 0.75781250, require_grad = True
module.blocks.18.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.18.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.mlp.w1.weight: torch.Size([1536, 512]) -0.00003487, 0.04907227, 0.60205078, require_grad = True
module.blocks.18.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.mlp.w2.weight: torch.Size([1536, 512]) 0.00011718, 0.04904175, 0.57470703, require_grad = True
module.blocks.18.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.18.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.18.mlp.w3.weight: torch.Size([512, 1536]) -0.00019193, 0.04904175, 0.63281250, require_grad = True
module.blocks.18.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.19.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.19.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.19.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.attn.q_proj.weight: torch.Size([512, 512]) -0.00011557, 0.03530884, 0.39550781, require_grad = True
module.blocks.19.attn.k_proj.weight: torch.Size([512, 512]) -0.00001448, 0.03521729, 0.40258789, require_grad = True
module.blocks.19.attn.v_proj.weight: torch.Size([512, 512]) 0.00020003, 0.06939697, 0.82031250, require_grad = True
module.blocks.19.attn.proj.weight: torch.Size([512, 512]) -0.00007510, 0.06921387, 0.75244141, require_grad = True
module.blocks.19.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.19.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.mlp.w1.weight: torch.Size([1536, 512]) -0.00009060, 0.04901123, 0.58105469, require_grad = True
module.blocks.19.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.mlp.w2.weight: torch.Size([1536, 512]) 0.00005192, 0.04916382, 0.57910156, require_grad = True
module.blocks.19.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.19.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.19.mlp.w3.weight: torch.Size([512, 1536]) 0.00014448, 0.04904175, 0.58740234, require_grad = True
module.blocks.19.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.20.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.20.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.20.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.attn.q_proj.weight: torch.Size([512, 512]) -0.00001329, 0.03533936, 0.42431641, require_grad = True
module.blocks.20.attn.k_proj.weight: torch.Size([512, 512]) 0.00008190, 0.03533936, 0.38037109, require_grad = True
module.blocks.20.attn.v_proj.weight: torch.Size([512, 512]) -0.00008249, 0.06958008, 0.80957031, require_grad = True
module.blocks.20.attn.proj.weight: torch.Size([512, 512]) -0.00022054, 0.06933594, 0.77978516, require_grad = True
module.blocks.20.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.20.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.mlp.w1.weight: torch.Size([1536, 512]) -0.00001764, 0.04904175, 0.55224609, require_grad = True
module.blocks.20.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.mlp.w2.weight: torch.Size([1536, 512]) -0.00011927, 0.04904175, 0.62255859, require_grad = True
module.blocks.20.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.20.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.20.mlp.w3.weight: torch.Size([512, 1536]) -0.00000018, 0.04904175, 0.61718750, require_grad = True
module.blocks.20.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.21.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.21.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.21.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.attn.q_proj.weight: torch.Size([512, 512]) -0.00006521, 0.03524780, 0.40917969, require_grad = True
module.blocks.21.attn.k_proj.weight: torch.Size([512, 512]) -0.00006926, 0.03530884, 0.40087891, require_grad = True
module.blocks.21.attn.v_proj.weight: torch.Size([512, 512]) -0.00015736, 0.06939697, 0.81738281, require_grad = True
module.blocks.21.attn.proj.weight: torch.Size([512, 512]) -0.00021255, 0.06933594, 0.79150391, require_grad = True
module.blocks.21.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.21.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.mlp.w1.weight: torch.Size([1536, 512]) -0.00000548, 0.04910278, 0.60156250, require_grad = True
module.blocks.21.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.mlp.w2.weight: torch.Size([1536, 512]) -0.00006860, 0.04913330, 0.58496094, require_grad = True
module.blocks.21.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.21.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.21.mlp.w3.weight: torch.Size([512, 1536]) 0.00009227, 0.04907227, 0.65136719, require_grad = True
module.blocks.21.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.22.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.22.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.22.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.attn.q_proj.weight: torch.Size([512, 512]) -0.00006175, 0.03518677, 0.41406250, require_grad = True
module.blocks.22.attn.k_proj.weight: torch.Size([512, 512]) -0.00005287, 0.03521729, 0.42871094, require_grad = True
module.blocks.22.attn.v_proj.weight: torch.Size([512, 512]) -0.00026250, 0.06939697, 0.79980469, require_grad = True
module.blocks.22.attn.proj.weight: torch.Size([512, 512]) -0.00008953, 0.06945801, 0.80371094, require_grad = True
module.blocks.22.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.22.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.mlp.w1.weight: torch.Size([1536, 512]) -0.00003433, 0.04907227, 0.63769531, require_grad = True
module.blocks.22.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.mlp.w2.weight: torch.Size([1536, 512]) -0.00001830, 0.04907227, 0.59277344, require_grad = True
module.blocks.22.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.22.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.22.mlp.w3.weight: torch.Size([512, 1536]) -0.00010282, 0.04904175, 0.62792969, require_grad = True
module.blocks.22.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.23.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.23.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.23.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.attn.q_proj.weight: torch.Size([512, 512]) -0.00011230, 0.03527832, 0.39208984, require_grad = True
module.blocks.23.attn.k_proj.weight: torch.Size([512, 512]) 0.00005651, 0.03524780, 0.38818359, require_grad = True
module.blocks.23.attn.v_proj.weight: torch.Size([512, 512]) 0.00024033, 0.06927490, 0.79785156, require_grad = True
module.blocks.23.attn.proj.weight: torch.Size([512, 512]) 0.00000584, 0.06939697, 0.79980469, require_grad = True
module.blocks.23.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.23.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.mlp.w1.weight: torch.Size([1536, 512]) -0.00000083, 0.04916382, 0.61132812, require_grad = True
module.blocks.23.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.mlp.w2.weight: torch.Size([1536, 512]) 0.00004011, 0.04907227, 0.59667969, require_grad = True
module.blocks.23.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.23.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.23.mlp.w3.weight: torch.Size([512, 1536]) -0.00002581, 0.04904175, 0.57617188, require_grad = True
module.blocks.23.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.24.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.24.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.24.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.attn.q_proj.weight: torch.Size([512, 512]) -0.00000137, 0.03530884, 0.40917969, require_grad = True
module.blocks.24.attn.k_proj.weight: torch.Size([512, 512]) 0.00008106, 0.03536987, 0.41601562, require_grad = True
module.blocks.24.attn.v_proj.weight: torch.Size([512, 512]) -0.00015557, 0.06945801, 0.78564453, require_grad = True
module.blocks.24.attn.proj.weight: torch.Size([512, 512]) -0.00016832, 0.06915283, 0.81396484, require_grad = True
module.blocks.24.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.24.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.mlp.w1.weight: torch.Size([1536, 512]) 0.00004995, 0.04907227, 0.58105469, require_grad = True
module.blocks.24.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.mlp.w2.weight: torch.Size([1536, 512]) -0.00005877, 0.04907227, 0.56201172, require_grad = True
module.blocks.24.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.24.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.24.mlp.w3.weight: torch.Size([512, 1536]) 0.00002980, 0.04910278, 0.58789062, require_grad = True
module.blocks.24.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.25.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.25.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.25.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.attn.q_proj.weight: torch.Size([512, 512]) -0.00007463, 0.03524780, 0.42407227, require_grad = True
module.blocks.25.attn.k_proj.weight: torch.Size([512, 512]) 0.00007856, 0.03524780, 0.41406250, require_grad = True
module.blocks.25.attn.v_proj.weight: torch.Size([512, 512]) 0.00017285, 0.06945801, 0.76025391, require_grad = True
module.blocks.25.attn.proj.weight: torch.Size([512, 512]) 0.00006133, 0.06945801, 0.79541016, require_grad = True
module.blocks.25.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.25.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.mlp.w1.weight: torch.Size([1536, 512]) 0.00001389, 0.04904175, 0.56982422, require_grad = True
module.blocks.25.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.mlp.w2.weight: torch.Size([1536, 512]) 0.00003338, 0.04910278, 0.59667969, require_grad = True
module.blocks.25.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.25.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.25.mlp.w3.weight: torch.Size([512, 1536]) 0.00001878, 0.04904175, 0.58105469, require_grad = True
module.blocks.25.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.26.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.26.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.26.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.attn.q_proj.weight: torch.Size([512, 512]) -0.00017715, 0.03530884, 0.40771484, require_grad = True
module.blocks.26.attn.k_proj.weight: torch.Size([512, 512]) -0.00000089, 0.03533936, 0.40771484, require_grad = True
module.blocks.26.attn.v_proj.weight: torch.Size([512, 512]) -0.00004077, 0.06945801, 0.80371094, require_grad = True
module.blocks.26.attn.proj.weight: torch.Size([512, 512]) -0.00001764, 0.06927490, 0.74218750, require_grad = True
module.blocks.26.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.26.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.mlp.w1.weight: torch.Size([1536, 512]) 0.00005037, 0.04907227, 0.55371094, require_grad = True
module.blocks.26.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.mlp.w2.weight: torch.Size([1536, 512]) -0.00009811, 0.04898071, 0.58300781, require_grad = True
module.blocks.26.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.26.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.26.mlp.w3.weight: torch.Size([512, 1536]) -0.00011235, 0.04901123, 0.57812500, require_grad = True
module.blocks.26.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.27.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.27.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.27.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.attn.q_proj.weight: torch.Size([512, 512]) -0.00009251, 0.03524780, 0.43945312, require_grad = True
module.blocks.27.attn.k_proj.weight: torch.Size([512, 512]) 0.00013268, 0.03527832, 0.38867188, require_grad = True
module.blocks.27.attn.v_proj.weight: torch.Size([512, 512]) -0.00017786, 0.06951904, 0.79833984, require_grad = True
module.blocks.27.attn.proj.weight: torch.Size([512, 512]) -0.00024140, 0.06933594, 0.85693359, require_grad = True
module.blocks.27.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.27.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.mlp.w1.weight: torch.Size([1536, 512]) -0.00000864, 0.04901123, 0.60107422, require_grad = True
module.blocks.27.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.mlp.w2.weight: torch.Size([1536, 512]) -0.00003427, 0.04904175, 0.57958984, require_grad = True
module.blocks.27.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.27.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.27.mlp.w3.weight: torch.Size([512, 1536]) 0.00003827, 0.04904175, 0.56396484, require_grad = True
module.blocks.27.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.28.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.28.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.28.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.attn.q_proj.weight: torch.Size([512, 512]) 0.00002611, 0.03518677, 0.38989258, require_grad = True
module.blocks.28.attn.k_proj.weight: torch.Size([512, 512]) -0.00007993, 0.03527832, 0.40771484, require_grad = True
module.blocks.28.attn.v_proj.weight: torch.Size([512, 512]) 0.00005817, 0.06927490, 0.78125000, require_grad = True
module.blocks.28.attn.proj.weight: torch.Size([512, 512]) 0.00009823, 0.06939697, 0.80859375, require_grad = True
module.blocks.28.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.28.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.mlp.w1.weight: torch.Size([1536, 512]) 0.00000739, 0.04910278, 0.57714844, require_grad = True
module.blocks.28.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.mlp.w2.weight: torch.Size([1536, 512]) -0.00006169, 0.04904175, 0.56835938, require_grad = True
module.blocks.28.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.28.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.28.mlp.w3.weight: torch.Size([512, 1536]) 0.00000185, 0.04904175, 0.56445312, require_grad = True
module.blocks.28.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.29.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.29.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.29.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.attn.q_proj.weight: torch.Size([512, 512]) -0.00001758, 0.03527832, 0.38623047, require_grad = True
module.blocks.29.attn.k_proj.weight: torch.Size([512, 512]) -0.00004071, 0.03521729, 0.40380859, require_grad = True
module.blocks.29.attn.v_proj.weight: torch.Size([512, 512]) 0.00023580, 0.06927490, 0.82666016, require_grad = True
module.blocks.29.attn.proj.weight: torch.Size([512, 512]) -0.00007617, 0.06921387, 0.81152344, require_grad = True
module.blocks.29.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.29.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.mlp.w1.weight: torch.Size([1536, 512]) -0.00020063, 0.04898071, 0.56152344, require_grad = True
module.blocks.29.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.mlp.w2.weight: torch.Size([1536, 512]) 0.00006908, 0.04907227, 0.61816406, require_grad = True
module.blocks.29.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.29.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.29.mlp.w3.weight: torch.Size([512, 1536]) -0.00002891, 0.04904175, 0.61132812, require_grad = True
module.blocks.29.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.30.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.30.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.30.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.attn.q_proj.weight: torch.Size([512, 512]) 0.00002885, 0.03515625, 0.40673828, require_grad = True
module.blocks.30.attn.k_proj.weight: torch.Size([512, 512]) -0.00011742, 0.03527832, 0.42138672, require_grad = True
module.blocks.30.attn.v_proj.weight: torch.Size([512, 512]) -0.00018990, 0.06951904, 0.79882812, require_grad = True
module.blocks.30.attn.proj.weight: torch.Size([512, 512]) -0.00001645, 0.06927490, 0.77734375, require_grad = True
module.blocks.30.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.30.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.mlp.w1.weight: torch.Size([1536, 512]) 0.00002491, 0.04907227, 0.63037109, require_grad = True
module.blocks.30.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.mlp.w2.weight: torch.Size([1536, 512]) -0.00002742, 0.04904175, 0.61230469, require_grad = True
module.blocks.30.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.30.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.30.mlp.w3.weight: torch.Size([512, 1536]) 0.00006366, 0.04910278, 0.58496094, require_grad = True
module.blocks.30.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.gamma_1: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.31.gamma_2: torch.Size([512]) 0.09997559, 0.09997559, 0.00000000, require_grad = True
module.blocks.31.norm1.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.31.norm1.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.attn.q_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.attn.v_bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.attn.q_proj.weight: torch.Size([512, 512]) 0.00008702, 0.03518677, 0.40966797, require_grad = True
module.blocks.31.attn.k_proj.weight: torch.Size([512, 512]) 0.00009894, 0.03530884, 0.38183594, require_grad = True
module.blocks.31.attn.v_proj.weight: torch.Size([512, 512]) -0.00005919, 0.06933594, 0.74462891, require_grad = True
module.blocks.31.attn.proj.weight: torch.Size([512, 512]) -0.00037074, 0.06933594, 0.78515625, require_grad = True
module.blocks.31.attn.proj.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.norm2.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.31.norm2.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.mlp.w1.weight: torch.Size([1536, 512]) -0.00004989, 0.04904175, 0.61767578, require_grad = True
module.blocks.31.mlp.w1.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.mlp.w2.weight: torch.Size([1536, 512]) 0.00000888, 0.04910278, 0.61132812, require_grad = True
module.blocks.31.mlp.w2.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.mlp.ffn_ln.weight: torch.Size([1536]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.blocks.31.mlp.ffn_ln.bias: torch.Size([1536]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.blocks.31.mlp.w3.weight: torch.Size([512, 1536]) 0.00000274, 0.04904175, 0.57031250, require_grad = True
module.blocks.31.mlp.w3.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.align_dim_16tofpn.weight: torch.Size([256, 512, 1, 1]) 0.00008881, 0.02207947, 0.08837891, require_grad = True
module.align_dim_16tofpn.bias: torch.Size([256]) 0.00003439, 0.02217102, 0.08807373, require_grad = True
module.fpn_modules.0.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.fpn_modules.0.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.fpn_modules.0.mlp.fc1.weight: torch.Size([682, 256, 1, 1]) 0.00007647, 0.06143188, 0.24597168, require_grad = True
module.fpn_modules.0.mlp.fc1.bias: torch.Size([682]) 0.00474167, 0.06188965, 0.24536133, require_grad = True
module.fpn_modules.0.mlp.fc2.weight: torch.Size([256, 682, 1, 1]) 0.00003314, 0.03775024, 0.15063477, require_grad = True
module.fpn_modules.0.mlp.fc2.bias: torch.Size([256]) 0.00085115, 0.03765869, 0.15051270, require_grad = True
module.fpn_modules.1.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.fpn_modules.1.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.fpn_modules.1.mlp.fc1.weight: torch.Size([682, 256, 1, 1]) 0.00026321, 0.06149292, 0.24597168, require_grad = True
module.fpn_modules.1.mlp.fc1.bias: torch.Size([682]) 0.00235558, 0.05929565, 0.24499512, require_grad = True
module.fpn_modules.1.mlp.fc2.weight: torch.Size([256, 682, 1, 1]) -0.00005037, 0.03762817, 0.15063477, require_grad = True
module.fpn_modules.1.mlp.fc2.bias: torch.Size([256]) 0.00238419, 0.03671265, 0.15026855, require_grad = True
module.fpn_modules.2.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.fpn_modules.2.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.fpn_modules.2.mlp.fc1.weight: torch.Size([682, 256, 1, 1]) -0.00021887, 0.06143188, 0.24597168, require_grad = True
module.fpn_modules.2.mlp.fc1.bias: torch.Size([682]) -0.00042224, 0.06015015, 0.24536133, require_grad = True
module.fpn_modules.2.mlp.fc2.weight: torch.Size([256, 682, 1, 1]) 0.00000125, 0.03771973, 0.15063477, require_grad = True
module.fpn_modules.2.mlp.fc2.bias: torch.Size([256]) -0.00531387, 0.03833008, 0.14953613, require_grad = True
module.align_dim_16to8.weight: torch.Size([256, 256, 1, 1]) 0.00010234, 0.03117371, 0.12500000, require_grad = True
module.align_dim_16to8.bias: torch.Size([256]) -0.00105667, 0.03085327, 0.12463379, require_grad = True
module.split_16to8.norm.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.split_16to8.norm.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.split_16to8.reduction.weight: torch.Size([512, 256, 2, 2]) 0.00001490, 0.01562500, 0.06250000, require_grad = True
module.split_16to8.reduction.bias: torch.Size([256]) -0.00055218, 0.01490784, 0.06210327, require_grad = True
module.block_16to8.0.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.block_16to8.0.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.block_16to8.0.mlp.fc1.weight: torch.Size([682, 256, 1, 1]) 0.00003397, 0.06140137, 0.24597168, require_grad = True
module.block_16to8.0.mlp.fc1.bias: torch.Size([682]) -0.00101757, 0.06002808, 0.24414062, require_grad = True
module.block_16to8.0.mlp.fc2.weight: torch.Size([256, 682, 1, 1]) -0.00015235, 0.03768921, 0.15063477, require_grad = True
module.block_16to8.0.mlp.fc2.bias: torch.Size([256]) -0.00408554, 0.03912354, 0.14953613, require_grad = True
module.align_dim_8to4.weight: torch.Size([256, 128, 1, 1]) -0.00076199, 0.04449463, 0.17675781, require_grad = True
module.align_dim_8to4.bias: torch.Size([256]) -0.00365639, 0.04412842, 0.17553711, require_grad = True
module.split_8to4.norm.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.split_8to4.norm.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.split_8to4.reduction.weight: torch.Size([256, 256, 2, 2]) -0.00004637, 0.01564026, 0.06250000, require_grad = True
module.split_8to4.reduction.bias: torch.Size([256]) -0.00105190, 0.01470184, 0.06222534, require_grad = True
module.block_8to4.0.norm2.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.block_8to4.0.norm2.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.block_8to4.0.mlp.fc1.weight: torch.Size([682, 256, 1, 1]) 0.00028706, 0.06146240, 0.24597168, require_grad = True
module.block_8to4.0.mlp.fc1.bias: torch.Size([682]) 0.00048137, 0.06094360, 0.24511719, require_grad = True
module.block_8to4.0.mlp.fc2.weight: torch.Size([256, 682, 1, 1]) 0.00012350, 0.03768921, 0.15063477, require_grad = True
module.block_8to4.0.mlp.fc2.bias: torch.Size([256]) 0.00024819, 0.03707886, 0.14501953, require_grad = True
module.decoder_embed.0.0.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.decoder_embed.0.0.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.decoder_embed.0.1.weight: torch.Size([512, 256, 1, 1]) -0.00018990, 0.03121948, 0.12500000, require_grad = True
module.decoder_embed.0.1.bias: torch.Size([512]) 0.00072575, 0.03247070, 0.12487793, require_grad = True
module.decoder_embed.1.0.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.decoder_embed.1.0.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.decoder_embed.1.1.weight: torch.Size([512, 256, 2, 2]) 0.00001907, 0.01560974, 0.06250000, require_grad = True
module.decoder_embed.1.1.bias: torch.Size([512]) 0.00008637, 0.01530457, 0.06231689, require_grad = True
module.decoder_embed.2.0.weight: torch.Size([256]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.decoder_embed.2.0.bias: torch.Size([256]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.decoder_embed.2.1.weight: torch.Size([512, 256, 4, 4]) 0.00001478, 0.00781250, 0.03125000, require_grad = True
module.decoder_embed.2.1.bias: torch.Size([512]) 0.00014329, 0.00747299, 0.03103638, require_grad = True
module.norm.weight: torch.Size([512]) 1.00000000, 1.00000000, 0.00000000, require_grad = True
module.norm.bias: torch.Size([512]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
module.lm_head.weight: torch.Size([768, 512]) 0.00001901, 0.03152466, 0.35791016, require_grad = True
module.lm_head.bias: torch.Size([768]) 0.00000000, 0.00000000, 0.00000000, require_grad = True
Start training for 300 epochs
[2024-07-24 02:13:26,892] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.86 | optimizer_gradients: 4.91 | optimizer_step: 1.42
[2024-07-24 02:13:26,893] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 361.82 | bwd_microstep: 661.53 | bwd_inner_microstep: 569.76 | bwd_allreduce_microstep: 91.67 | step_microstep: 100.98
[2024-07-24 02:13:26,893] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 361.83 | bwd: 661.51 | bwd_inner: 569.79 | bwd_allreduce: 91.66 | step: 100.99
Epoch: [0]  [    0/10009]  eta: 5:41:17  lr: 0.000000  min_lr: 0.000000  all_loss_mean: 0.0364 (0.0364)  loss: 0.0383 (0.0383)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 2.0460  data: 0.5882  max mem: 9816
[2024-07-24 02:13:27,170] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.85 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:27,170] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 54.11 | bwd_microstep: 146.45 | bwd_inner_microstep: 60.66 | bwd_allreduce_microstep: 85.76 | step_microstep: 57.38
[2024-07-24 02:13:27,170] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 54.11 | bwd: 146.44 | bwd_inner: 60.65 | bwd_allreduce: 85.76 | step: 57.39
[2024-07-24 02:13:27,423] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.87 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:13:27,423] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 145.98 | bwd_inner_microstep: 60.15 | bwd_allreduce_microstep: 85.80 | step_microstep: 57.48
[2024-07-24 02:13:27,423] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.97 | bwd_inner: 60.14 | bwd_allreduce: 85.80 | step: 57.49
[2024-07-24 02:13:27,676] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.47 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:27,676] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 145.78 | bwd_inner_microstep: 59.75 | bwd_allreduce_microstep: 85.99 | step_microstep: 57.11
[2024-07-24 02:13:27,676] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.76 | bwd_inner: 59.74 | bwd_allreduce: 85.99 | step: 57.11
[2024-07-24 02:13:27,929] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.12 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:27,929] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 145.43 | bwd_inner_microstep: 59.66 | bwd_allreduce_microstep: 85.74 | step_microstep: 57.69
[2024-07-24 02:13:27,929] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 145.42 | bwd_inner: 59.65 | bwd_allreduce: 85.74 | step: 57.70
[2024-07-24 02:13:28,184] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.78 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:28,184] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 32.61 | bwd_microstep: 145.66 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 85.69 | step_microstep: 57.87
[2024-07-24 02:13:28,185] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 32.62 | bwd: 145.65 | bwd_inner: 59.93 | bwd_allreduce: 85.69 | step: 57.88
[2024-07-24 02:13:28,436] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.56 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:28,437] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.35 | bwd_inner_microstep: 59.75 | bwd_allreduce_microstep: 85.57 | step_microstep: 57.40
[2024-07-24 02:13:28,437] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.34 | bwd_inner: 59.74 | bwd_allreduce: 85.57 | step: 57.41
[2024-07-24 02:13:28,698] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.71 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:28,699] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.42 | bwd_inner_microstep: 60.52 | bwd_allreduce_microstep: 85.87 | step_microstep: 57.43
[2024-07-24 02:13:28,699] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.13 | bwd: 146.40 | bwd_inner: 60.51 | bwd_allreduce: 85.87 | step: 57.44
[2024-07-24 02:13:28,954] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.86 | optimizer_gradients: 0.65 | optimizer_step: 0.92
[2024-07-24 02:13:28,955] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 146.46 | bwd_inner_microstep: 60.24 | bwd_allreduce_microstep: 86.19 | step_microstep: 59.30
[2024-07-24 02:13:28,955] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 146.45 | bwd_inner: 60.23 | bwd_allreduce: 86.19 | step: 59.31
[2024-07-24 02:13:29,209] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:29,209] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.90 | bwd_inner_microstep: 60.29 | bwd_allreduce_microstep: 86.58 | step_microstep: 57.52
[2024-07-24 02:13:29,209] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.89 | bwd_inner: 60.28 | bwd_allreduce: 86.58 | step: 57.53
[2024-07-24 02:13:29,463] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:29,463] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.23 | bwd_inner_microstep: 60.02 | bwd_allreduce_microstep: 86.17 | step_microstep: 57.73
[2024-07-24 02:13:29,463] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.21 | bwd_inner: 60.01 | bwd_allreduce: 86.17 | step: 57.75
Epoch: [0]  [   10/10009]  eta: 1:09:55  lr: 0.000000  min_lr: 0.000000  all_loss_mean: 0.0388 (0.0395)  loss: 0.0426 (0.0413)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.4195  data: 0.0536  max mem: 9933
[2024-07-24 02:13:29,718] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.60 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:29,718] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.27 | bwd_inner_microstep: 60.43 | bwd_allreduce_microstep: 85.80 | step_microstep: 57.17
[2024-07-24 02:13:29,718] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.26 | bwd_inner: 60.42 | bwd_allreduce: 85.80 | step: 57.19
[2024-07-24 02:13:29,971] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.09 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:29,972] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.16 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 86.19 | step_microstep: 57.62
[2024-07-24 02:13:29,972] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.14 | bwd_inner: 59.93 | bwd_allreduce: 86.19 | step: 57.64
[2024-07-24 02:13:30,225] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.69 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:30,225] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 145.96 | bwd_inner_microstep: 60.32 | bwd_allreduce_microstep: 85.60 | step_microstep: 57.26
[2024-07-24 02:13:30,225] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.95 | bwd_inner: 60.31 | bwd_allreduce: 85.60 | step: 57.27
[2024-07-24 02:13:30,480] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:30,481] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 32.74 | bwd_microstep: 145.93 | bwd_inner_microstep: 60.34 | bwd_allreduce_microstep: 85.55 | step_microstep: 57.57
[2024-07-24 02:13:30,481] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 32.75 | bwd: 145.92 | bwd_inner: 60.33 | bwd_allreduce: 85.55 | step: 57.58
[2024-07-24 02:13:30,733] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.46 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:30,734] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.35 | bwd_microstep: 145.89 | bwd_inner_microstep: 60.06 | bwd_allreduce_microstep: 85.79 | step_microstep: 57.08
[2024-07-24 02:13:30,734] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.35 | bwd: 145.87 | bwd_inner: 60.05 | bwd_allreduce: 85.79 | step: 57.10
[2024-07-24 02:13:30,987] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.15 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:30,988] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.18 | bwd_inner_microstep: 59.92 | bwd_allreduce_microstep: 86.23 | step_microstep: 58.33
[2024-07-24 02:13:30,989] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 146.17 | bwd_inner: 59.91 | bwd_allreduce: 86.23 | step: 58.36
[2024-07-24 02:13:31,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.14 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:31,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 40.34 | bwd_microstep: 145.68 | bwd_inner_microstep: 59.69 | bwd_allreduce_microstep: 85.96 | step_microstep: 57.78
[2024-07-24 02:13:31,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 40.34 | bwd: 145.67 | bwd_inner: 59.68 | bwd_allreduce: 85.96 | step: 57.79
[2024-07-24 02:13:31,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:31,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.26 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 85.99 | step_microstep: 57.54
[2024-07-24 02:13:31,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.13 | bwd: 146.24 | bwd_inner: 60.23 | bwd_allreduce: 85.99 | step: 57.55
[2024-07-24 02:13:31,759] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:31,759] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.93 | bwd_microstep: 146.07 | bwd_inner_microstep: 59.86 | bwd_allreduce_microstep: 86.17 | step_microstep: 57.45
[2024-07-24 02:13:31,759] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.94 | bwd: 146.05 | bwd_inner: 59.85 | bwd_allreduce: 86.17 | step: 57.46
[2024-07-24 02:13:32,013] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.16 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:32,013] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.99 | bwd_inner_microstep: 60.12 | bwd_allreduce_microstep: 85.84 | step_microstep: 58.17
[2024-07-24 02:13:32,013] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 145.97 | bwd_inner: 60.11 | bwd_allreduce: 85.84 | step: 58.18
Epoch: [0]  [   20/10009]  eta: 0:56:47  lr: 0.000000  min_lr: 0.000000  all_loss_mean: 0.0382 (0.0386)  loss: 0.0394 (0.0392)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2559  data: 0.0002  max mem: 9933
[2024-07-24 02:13:32,268] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.05 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:13:32,268] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.54 | bwd_inner_microstep: 59.81 | bwd_allreduce_microstep: 86.70 | step_microstep: 57.74
[2024-07-24 02:13:32,268] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.53 | bwd_inner: 59.80 | bwd_allreduce: 86.70 | step: 57.75
[2024-07-24 02:13:32,522] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.71 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:32,523] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 147.57 | bwd_inner_microstep: 59.99 | bwd_allreduce_microstep: 87.55 | step_microstep: 57.31
[2024-07-24 02:13:32,523] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 147.56 | bwd_inner: 59.98 | bwd_allreduce: 87.55 | step: 57.32
[2024-07-24 02:13:32,775] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.69 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:32,776] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 145.39 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 85.25 | step_microstep: 57.55
[2024-07-24 02:13:32,776] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.38 | bwd_inner: 60.10 | bwd_allreduce: 85.25 | step: 57.56
[2024-07-24 02:13:33,029] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.55 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:33,030] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 145.93 | bwd_inner_microstep: 60.26 | bwd_allreduce_microstep: 85.64 | step_microstep: 57.21
[2024-07-24 02:13:33,030] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.92 | bwd_inner: 60.25 | bwd_allreduce: 85.64 | step: 57.22
[2024-07-24 02:13:33,283] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.68 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:33,284] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 146.43 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 86.29 | step_microstep: 57.30
[2024-07-24 02:13:33,284] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 146.41 | bwd_inner: 60.10 | bwd_allreduce: 86.29 | step: 57.31
[2024-07-24 02:13:33,537] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.32 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:33,538] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 145.94 | bwd_inner_microstep: 60.34 | bwd_allreduce_microstep: 85.57 | step_microstep: 56.93
[2024-07-24 02:13:33,538] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 145.93 | bwd_inner: 60.33 | bwd_allreduce: 85.57 | step: 56.94
[2024-07-24 02:13:33,798] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.65 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:33,798] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 145.44 | bwd_inner_microstep: 60.64 | bwd_allreduce_microstep: 84.77 | step_microstep: 57.75
[2024-07-24 02:13:33,798] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 145.43 | bwd_inner: 60.62 | bwd_allreduce: 84.77 | step: 57.76
[2024-07-24 02:13:34,051] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.68 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:34,051] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 145.88 | bwd_inner_microstep: 59.59 | bwd_allreduce_microstep: 86.26 | step_microstep: 57.35
[2024-07-24 02:13:34,051] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.87 | bwd_inner: 59.59 | bwd_allreduce: 86.26 | step: 57.36
[2024-07-24 02:13:34,305] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.51 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:34,305] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 146.66 | bwd_inner_microstep: 59.97 | bwd_allreduce_microstep: 86.65 | step_microstep: 57.17
[2024-07-24 02:13:34,305] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.64 | bwd_inner: 59.96 | bwd_allreduce: 86.65 | step: 57.19
[2024-07-24 02:13:34,558] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.64 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:34,558] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.07 | bwd_microstep: 146.50 | bwd_inner_microstep: 60.08 | bwd_allreduce_microstep: 86.39 | step_microstep: 57.56
[2024-07-24 02:13:34,559] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 146.49 | bwd_inner: 60.07 | bwd_allreduce: 86.39 | step: 57.57
Epoch: [0]  [   30/10009]  eta: 0:52:05  lr: 0.000000  min_lr: 0.000000  all_loss_mean: 0.0356 (0.0351)  loss: 0.0343 (0.0359)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2547  data: 0.0001  max mem: 9933
[2024-07-24 02:13:34,811] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.19 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:34,812] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.06 | bwd_inner_microstep: 60.10 | bwd_allreduce_microstep: 85.92 | step_microstep: 56.83
[2024-07-24 02:13:34,812] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.04 | bwd_inner: 60.09 | bwd_allreduce: 85.92 | step: 56.84
[2024-07-24 02:13:35,067] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.46 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:35,068] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 145.74 | bwd_inner_microstep: 60.37 | bwd_allreduce_microstep: 85.34 | step_microstep: 58.85
[2024-07-24 02:13:35,068] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 145.73 | bwd_inner: 60.36 | bwd_allreduce: 85.34 | step: 58.87
[2024-07-24 02:13:35,321] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.85 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:35,322] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 146.59 | bwd_inner_microstep: 60.40 | bwd_allreduce_microstep: 86.16 | step_microstep: 57.56
[2024-07-24 02:13:35,322] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 146.58 | bwd_inner: 60.40 | bwd_allreduce: 86.16 | step: 57.57
[2024-07-24 02:13:35,575] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.26 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:35,575] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.28 | bwd_microstep: 145.14 | bwd_inner_microstep: 60.62 | bwd_allreduce_microstep: 84.48 | step_microstep: 58.22
[2024-07-24 02:13:35,575] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.29 | bwd: 145.12 | bwd_inner: 60.61 | bwd_allreduce: 84.48 | step: 58.23
[2024-07-24 02:13:35,828] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.87 | optimizer_gradients: 0.56 | optimizer_step: 0.91
[2024-07-24 02:13:35,829] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 145.91 | bwd_inner_microstep: 60.29 | bwd_allreduce_microstep: 85.59 | step_microstep: 57.30
[2024-07-24 02:13:35,829] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 145.90 | bwd_inner: 60.28 | bwd_allreduce: 85.59 | step: 57.31
[2024-07-24 02:13:36,081] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.60 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:36,081] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.90 | bwd_inner_microstep: 59.86 | bwd_allreduce_microstep: 86.01 | step_microstep: 57.08
[2024-07-24 02:13:36,081] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.89 | bwd_inner: 59.86 | bwd_allreduce: 86.01 | step: 57.10
[2024-07-24 02:13:36,333] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.34 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:36,334] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.61 | bwd_inner_microstep: 60.32 | bwd_allreduce_microstep: 85.26 | step_microstep: 57.06
[2024-07-24 02:13:36,334] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.60 | bwd_inner: 60.31 | bwd_allreduce: 85.26 | step: 57.07
[2024-07-24 02:13:36,587] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.63 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:36,587] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.26 | bwd_microstep: 146.05 | bwd_inner_microstep: 60.33 | bwd_allreduce_microstep: 85.69 | step_microstep: 57.56
[2024-07-24 02:13:36,588] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.26 | bwd: 146.04 | bwd_inner: 60.33 | bwd_allreduce: 85.69 | step: 57.58
[2024-07-24 02:13:36,840] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.28 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:36,841] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 145.75 | bwd_inner_microstep: 59.65 | bwd_allreduce_microstep: 86.08 | step_microstep: 58.20
[2024-07-24 02:13:36,841] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.74 | bwd_inner: 59.64 | bwd_allreduce: 86.08 | step: 58.21
[2024-07-24 02:13:37,095] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.38 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:37,095] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.09 | bwd_microstep: 146.32 | bwd_inner_microstep: 59.65 | bwd_allreduce_microstep: 86.59 | step_microstep: 58.07
[2024-07-24 02:13:37,095] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.10 | bwd: 146.25 | bwd_inner: 59.64 | bwd_allreduce: 86.59 | step: 58.08
Epoch: [0]  [   40/10009]  eta: 0:49:37  lr: 0.000001  min_lr: 0.000001  all_loss_mean: 0.0226 (0.0286)  loss: 0.0232 (0.0294)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2540  data: 0.0002  max mem: 9933
[2024-07-24 02:13:37,349] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.02 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:37,349] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.18 | bwd_inner_microstep: 60.18 | bwd_allreduce_microstep: 85.97 | step_microstep: 57.65
[2024-07-24 02:13:37,349] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.17 | bwd_inner: 60.17 | bwd_allreduce: 85.97 | step: 57.66
[2024-07-24 02:13:37,604] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.24 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:37,604] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.17 | bwd_inner_microstep: 60.05 | bwd_allreduce_microstep: 86.09 | step_microstep: 59.02
[2024-07-24 02:13:37,604] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.16 | bwd_inner: 60.03 | bwd_allreduce: 86.09 | step: 59.04
[2024-07-24 02:13:37,857] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.24 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:37,858] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 145.89 | bwd_inner_microstep: 60.39 | bwd_allreduce_microstep: 85.47 | step_microstep: 57.86
[2024-07-24 02:13:37,858] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.88 | bwd_inner: 60.38 | bwd_allreduce: 85.47 | step: 57.87
[2024-07-24 02:13:38,112] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.69 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:13:38,113] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.32 | bwd_microstep: 146.34 | bwd_inner_microstep: 60.32 | bwd_allreduce_microstep: 85.99 | step_microstep: 58.32
[2024-07-24 02:13:38,113] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.31 | bwd: 146.33 | bwd_inner: 60.31 | bwd_allreduce: 85.99 | step: 58.33
[2024-07-24 02:13:38,365] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.91 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:38,366] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.27 | bwd_microstep: 145.49 | bwd_inner_microstep: 60.29 | bwd_allreduce_microstep: 85.17 | step_microstep: 57.49
[2024-07-24 02:13:38,366] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.28 | bwd: 145.48 | bwd_inner: 60.28 | bwd_allreduce: 85.17 | step: 57.50
[2024-07-24 02:13:38,618] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.88 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:38,618] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.30 | bwd_microstep: 145.07 | bwd_inner_microstep: 60.09 | bwd_allreduce_microstep: 84.95 | step_microstep: 57.39
[2024-07-24 02:13:38,618] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.30 | bwd: 145.06 | bwd_inner: 60.08 | bwd_allreduce: 84.95 | step: 57.40
[2024-07-24 02:13:38,873] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.88 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:38,873] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.27 | bwd_inner_microstep: 59.81 | bwd_allreduce_microstep: 86.40 | step_microstep: 58.74
[2024-07-24 02:13:38,873] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.13 | bwd: 146.23 | bwd_inner: 59.80 | bwd_allreduce: 86.40 | step: 58.75
[2024-07-24 02:13:39,128] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.49 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:39,128] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 146.69 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 86.55 | step_microstep: 57.16
[2024-07-24 02:13:39,128] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 146.68 | bwd_inner: 60.10 | bwd_allreduce: 86.55 | step: 57.17
[2024-07-24 02:13:39,381] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.05 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:39,382] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.22 | bwd_inner_microstep: 60.29 | bwd_allreduce_microstep: 85.90 | step_microstep: 57.73
[2024-07-24 02:13:39,382] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.21 | bwd_inner: 60.28 | bwd_allreduce: 85.90 | step: 57.74
[2024-07-24 02:13:39,644] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.00 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:39,644] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 39.37 | bwd_microstep: 146.18 | bwd_inner_microstep: 60.18 | bwd_allreduce_microstep: 85.96 | step_microstep: 57.70
[2024-07-24 02:13:39,644] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 39.38 | bwd: 146.16 | bwd_inner: 60.17 | bwd_allreduce: 85.95 | step: 57.71
Epoch: [0]  [   50/10009]  eta: 0:48:08  lr: 0.000001  min_lr: 0.000001  all_loss_mean: -0.0057 (0.0190)  loss: -0.0024 (0.0200)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2542  data: 0.0002  max mem: 9933
[2024-07-24 02:13:39,898] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.96 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:39,899] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 145.79 | bwd_inner_microstep: 60.14 | bwd_allreduce_microstep: 85.61 | step_microstep: 58.40
[2024-07-24 02:13:39,899] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 145.77 | bwd_inner: 60.14 | bwd_allreduce: 85.61 | step: 58.41
[2024-07-24 02:13:40,153] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.02 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:40,153] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 145.98 | bwd_inner_microstep: 60.13 | bwd_allreduce_microstep: 85.82 | step_microstep: 57.60
[2024-07-24 02:13:40,153] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 145.97 | bwd_inner: 60.12 | bwd_allreduce: 85.82 | step: 57.61
[2024-07-24 02:13:40,406] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:40,406] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.27 | bwd_inner_microstep: 60.04 | bwd_allreduce_microstep: 86.19 | step_microstep: 57.36
[2024-07-24 02:13:40,406] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.26 | bwd_inner: 60.04 | bwd_allreduce: 86.19 | step: 57.37
[2024-07-24 02:13:40,661] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.55 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:40,661] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 146.64 | bwd_inner_microstep: 60.04 | bwd_allreduce_microstep: 86.58 | step_microstep: 57.68
[2024-07-24 02:13:40,661] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 146.63 | bwd_inner: 60.03 | bwd_allreduce: 86.58 | step: 57.70
[2024-07-24 02:13:40,915] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.18 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:40,915] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.58 | bwd_inner_microstep: 60.29 | bwd_allreduce_microstep: 86.25 | step_microstep: 57.46
[2024-07-24 02:13:40,915] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.56 | bwd_inner: 60.28 | bwd_allreduce: 86.25 | step: 57.47
[2024-07-24 02:13:41,176] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.26 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:41,176] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 153.85 | bwd_inner_microstep: 60.20 | bwd_allreduce_microstep: 93.62 | step_microstep: 57.25
[2024-07-24 02:13:41,177] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 153.84 | bwd_inner: 60.20 | bwd_allreduce: 93.62 | step: 57.26
[2024-07-24 02:13:41,429] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.61 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:13:41,429] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 145.95 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 85.98 | step_microstep: 57.19
[2024-07-24 02:13:41,430] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.94 | bwd_inner: 59.94 | bwd_allreduce: 85.98 | step: 57.20
[2024-07-24 02:13:41,683] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.95 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:13:41,683] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 145.48 | bwd_inner_microstep: 60.10 | bwd_allreduce_microstep: 85.35 | step_microstep: 57.67
[2024-07-24 02:13:41,683] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.47 | bwd_inner: 60.09 | bwd_allreduce: 85.35 | step: 57.68
[2024-07-24 02:13:41,936] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.12 | optimizer_gradients: 0.61 | optimizer_step: 0.92
[2024-07-24 02:13:41,936] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.62 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 85.36 | step_microstep: 57.91
[2024-07-24 02:13:41,937] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.61 | bwd_inner: 60.22 | bwd_allreduce: 85.36 | step: 57.92
[2024-07-24 02:13:42,190] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.94 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:42,191] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.89 | bwd_inner_microstep: 59.49 | bwd_allreduce_microstep: 87.38 | step_microstep: 57.62
[2024-07-24 02:13:42,191] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.88 | bwd_inner: 59.48 | bwd_allreduce: 87.38 | step: 57.63
Epoch: [0]  [   60/10009]  eta: 0:47:07  lr: 0.000001  min_lr: 0.000001  all_loss_mean: -0.0435 (0.0065)  loss: -0.0438 (0.0072)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2547  data: 0.0001  max mem: 9933
[2024-07-24 02:13:42,444] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.61 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:42,444] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 145.78 | bwd_inner_microstep: 59.90 | bwd_allreduce_microstep: 85.81 | step_microstep: 57.31
[2024-07-24 02:13:42,444] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 145.77 | bwd_inner: 59.89 | bwd_allreduce: 85.81 | step: 57.32
[2024-07-24 02:13:42,696] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.38 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:42,697] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 145.10 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 84.83 | step_microstep: 57.42
[2024-07-24 02:13:42,697] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.08 | bwd_inner: 60.23 | bwd_allreduce: 84.83 | step: 57.43
[2024-07-24 02:13:42,949] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.72 | optimizer_gradients: 0.60 | optimizer_step: 0.91
[2024-07-24 02:13:42,950] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.00 | bwd_inner_microstep: 59.90 | bwd_allreduce_microstep: 86.07 | step_microstep: 57.66
[2024-07-24 02:13:42,950] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 145.98 | bwd_inner: 59.89 | bwd_allreduce: 86.07 | step: 57.68
[2024-07-24 02:13:43,204] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.99 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:43,204] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.83 | bwd_inner_microstep: 60.29 | bwd_allreduce_microstep: 86.50 | step_microstep: 57.50
[2024-07-24 02:13:43,204] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.81 | bwd_inner: 60.28 | bwd_allreduce: 86.50 | step: 57.51
[2024-07-24 02:13:43,460] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.47 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:43,460] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 148.13 | bwd_inner_microstep: 60.46 | bwd_allreduce_microstep: 87.63 | step_microstep: 57.41
[2024-07-24 02:13:43,460] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 148.11 | bwd_inner: 60.45 | bwd_allreduce: 87.63 | step: 57.42
[2024-07-24 02:13:43,715] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.96 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:43,715] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 146.27 | bwd_inner_microstep: 60.13 | bwd_allreduce_microstep: 86.11 | step_microstep: 57.63
[2024-07-24 02:13:43,715] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.15 | bwd: 146.26 | bwd_inner: 60.12 | bwd_allreduce: 86.11 | step: 57.65
[2024-07-24 02:13:43,969] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.21 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:43,969] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.44 | bwd_inner_microstep: 60.01 | bwd_allreduce_microstep: 86.39 | step_microstep: 57.82
[2024-07-24 02:13:43,969] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.43 | bwd_inner: 60.00 | bwd_allreduce: 86.39 | step: 57.83
[2024-07-24 02:13:44,223] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.70 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:44,223] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.74 | bwd_inner_microstep: 60.21 | bwd_allreduce_microstep: 86.50 | step_microstep: 57.49
[2024-07-24 02:13:44,223] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.73 | bwd_inner: 60.20 | bwd_allreduce: 86.50 | step: 57.50
[2024-07-24 02:13:44,476] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.71 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:44,477] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 146.42 | bwd_inner_microstep: 60.21 | bwd_allreduce_microstep: 86.18 | step_microstep: 57.47
[2024-07-24 02:13:44,477] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.41 | bwd_inner: 60.21 | bwd_allreduce: 86.18 | step: 57.48
[2024-07-24 02:13:44,730] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.03 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:44,731] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.50 | bwd_inner_microstep: 60.51 | bwd_allreduce_microstep: 85.96 | step_microstep: 57.70
[2024-07-24 02:13:44,731] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.49 | bwd_inner: 60.51 | bwd_allreduce: 85.96 | step: 57.71
Epoch: [0]  [   70/10009]  eta: 0:46:22  lr: 0.000001  min_lr: 0.000001  all_loss_mean: -0.0756 (-0.0084)  loss: -0.0743 (-0.0077)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2542  data: 0.0001  max mem: 9933
[2024-07-24 02:13:44,984] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.46 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:44,984] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.90 | bwd_inner_microstep: 59.98 | bwd_allreduce_microstep: 85.89 | step_microstep: 57.40
[2024-07-24 02:13:44,984] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 145.89 | bwd_inner: 59.97 | bwd_allreduce: 85.89 | step: 57.41
[2024-07-24 02:13:45,237] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:45,237] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 145.81 | bwd_inner_microstep: 59.97 | bwd_allreduce_microstep: 85.81 | step_microstep: 57.48
[2024-07-24 02:13:45,238] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 145.80 | bwd_inner: 59.96 | bwd_allreduce: 85.81 | step: 57.49
[2024-07-24 02:13:45,491] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.99 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:45,491] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.06 | bwd_inner_microstep: 60.08 | bwd_allreduce_microstep: 85.95 | step_microstep: 58.03
[2024-07-24 02:13:45,491] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.05 | bwd_inner: 60.08 | bwd_allreduce: 85.95 | step: 58.04
[2024-07-24 02:13:45,744] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.55 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:45,744] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.07 | bwd_microstep: 146.34 | bwd_inner_microstep: 59.77 | bwd_allreduce_microstep: 86.55 | step_microstep: 57.12
[2024-07-24 02:13:45,744] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 146.33 | bwd_inner: 59.76 | bwd_allreduce: 86.55 | step: 57.13
[2024-07-24 02:13:45,997] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.51 | optimizer_gradients: 0.56 | optimizer_step: 0.91
[2024-07-24 02:13:45,998] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.06 | bwd_microstep: 146.59 | bwd_inner_microstep: 59.70 | bwd_allreduce_microstep: 86.86 | step_microstep: 57.14
[2024-07-24 02:13:45,998] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 146.58 | bwd_inner: 59.69 | bwd_allreduce: 86.86 | step: 57.15
[2024-07-24 02:13:46,250] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.77 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:46,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.08 | bwd_microstep: 146.31 | bwd_inner_microstep: 59.72 | bwd_allreduce_microstep: 86.56 | step_microstep: 57.35
[2024-07-24 02:13:46,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.09 | bwd: 146.30 | bwd_inner: 59.71 | bwd_allreduce: 86.56 | step: 57.37
[2024-07-24 02:13:46,504] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.06 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:46,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 146.21 | bwd_inner_microstep: 60.70 | bwd_allreduce_microstep: 85.48 | step_microstep: 57.59
[2024-07-24 02:13:46,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.20 | bwd_inner: 60.69 | bwd_allreduce: 85.48 | step: 57.61
[2024-07-24 02:13:46,759] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.78 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:46,759] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.38 | bwd_inner_microstep: 59.78 | bwd_allreduce_microstep: 86.57 | step_microstep: 57.45
[2024-07-24 02:13:46,759] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.37 | bwd_inner: 59.77 | bwd_allreduce: 86.57 | step: 57.46
/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/PIL/TiffImagePlugin.py:900: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.
  warnings.warn(str(msg))
[2024-07-24 02:13:47,012] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.91 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:47,013] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 145.92 | bwd_inner_microstep: 59.81 | bwd_allreduce_microstep: 86.08 | step_microstep: 57.52
[2024-07-24 02:13:47,013] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.91 | bwd_inner: 59.81 | bwd_allreduce: 86.08 | step: 57.53
[2024-07-24 02:13:47,267] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.88 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:47,268] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.09 | bwd_inner_microstep: 60.10 | bwd_allreduce_microstep: 85.96 | step_microstep: 57.75
[2024-07-24 02:13:47,268] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 146.08 | bwd_inner: 60.09 | bwd_allreduce: 85.96 | step: 57.76
Epoch: [0]  [   80/10009]  eta: 0:45:47  lr: 0.000001  min_lr: 0.000001  all_loss_mean: -0.1240 (-0.0249)  loss: -0.1229 (-0.0242)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2538  data: 0.0002  max mem: 9933
[2024-07-24 02:13:47,522] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:47,523] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.07 | bwd_microstep: 146.33 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 86.18 | step_microstep: 58.67
[2024-07-24 02:13:47,523] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 146.31 | bwd_inner: 60.10 | bwd_allreduce: 86.18 | step: 58.68
[2024-07-24 02:13:47,775] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.87 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:47,776] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 145.88 | bwd_inner_microstep: 59.96 | bwd_allreduce_microstep: 85.90 | step_microstep: 57.45
[2024-07-24 02:13:47,776] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 145.87 | bwd_inner: 59.95 | bwd_allreduce: 85.90 | step: 57.46
[2024-07-24 02:13:48,029] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:48,029] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.19 | bwd_inner_microstep: 60.35 | bwd_allreduce_microstep: 85.80 | step_microstep: 57.36
[2024-07-24 02:13:48,029] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.17 | bwd_inner: 60.35 | bwd_allreduce: 85.80 | step: 57.37
[2024-07-24 02:13:48,282] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.81 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:48,283] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.10 | bwd_microstep: 146.80 | bwd_inner_microstep: 59.60 | bwd_allreduce_microstep: 87.17 | step_microstep: 57.38
[2024-07-24 02:13:48,283] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.10 | bwd: 146.79 | bwd_inner: 59.59 | bwd_allreduce: 87.17 | step: 57.39
[2024-07-24 02:13:48,536] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.17 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:48,537] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.49 | bwd_inner_microstep: 60.16 | bwd_allreduce_microstep: 86.30 | step_microstep: 58.20
[2024-07-24 02:13:48,537] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.48 | bwd_inner: 60.15 | bwd_allreduce: 86.30 | step: 58.21
[2024-07-24 02:13:48,790] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.39 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:48,791] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.53 | bwd_inner_microstep: 60.22 | bwd_allreduce_microstep: 86.28 | step_microstep: 57.03
[2024-07-24 02:13:48,791] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 146.52 | bwd_inner: 60.20 | bwd_allreduce: 86.28 | step: 57.04
[2024-07-24 02:13:49,046] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.88 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:13:49,046] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 145.98 | bwd_inner_microstep: 60.43 | bwd_allreduce_microstep: 85.52 | step_microstep: 58.59
[2024-07-24 02:13:49,046] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.97 | bwd_inner: 60.43 | bwd_allreduce: 85.52 | step: 58.60
[2024-07-24 02:13:49,301] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:49,302] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.62 | bwd_inner_microstep: 60.14 | bwd_allreduce_microstep: 86.46 | step_microstep: 57.53
[2024-07-24 02:13:49,302] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.61 | bwd_inner: 60.13 | bwd_allreduce: 86.46 | step: 57.55
[2024-07-24 02:13:49,555] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.63 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:49,555] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.19 | bwd_inner_microstep: 60.31 | bwd_allreduce_microstep: 85.85 | step_microstep: 57.49
[2024-07-24 02:13:49,556] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.18 | bwd_inner: 60.30 | bwd_allreduce: 85.85 | step: 57.50
[2024-07-24 02:13:49,807] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.66 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:49,808] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 144.71 | bwd_inner_microstep: 59.82 | bwd_allreduce_microstep: 84.86 | step_microstep: 57.31
[2024-07-24 02:13:49,808] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 144.70 | bwd_inner: 59.81 | bwd_allreduce: 84.86 | step: 57.32
Epoch: [0]  [   90/10009]  eta: 0:45:19  lr: 0.000001  min_lr: 0.000001  all_loss_mean: -0.1655 (-0.0424)  loss: -0.1688 (-0.0417)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2538  data: 0.0002  max mem: 9933
[2024-07-24 02:13:50,061] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.69 | optimizer_gradients: 0.62 | optimizer_step: 0.91
[2024-07-24 02:13:50,062] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.10 | bwd_microstep: 146.58 | bwd_inner_microstep: 59.77 | bwd_allreduce_microstep: 86.78 | step_microstep: 57.40
[2024-07-24 02:13:50,062] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.10 | bwd: 146.57 | bwd_inner: 59.76 | bwd_allreduce: 86.78 | step: 57.41
[2024-07-24 02:13:50,314] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.64 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:50,314] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 145.77 | bwd_inner_microstep: 60.06 | bwd_allreduce_microstep: 85.68 | step_microstep: 57.21
[2024-07-24 02:13:50,315] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.76 | bwd_inner: 60.05 | bwd_allreduce: 85.68 | step: 57.22
[2024-07-24 02:13:50,568] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.24 | optimizer_gradients: 0.61 | optimizer_step: 0.91
[2024-07-24 02:13:50,568] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 145.75 | bwd_inner_microstep: 60.53 | bwd_allreduce_microstep: 85.19 | step_microstep: 58.03
[2024-07-24 02:13:50,568] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.74 | bwd_inner: 60.53 | bwd_allreduce: 85.19 | step: 58.04
[2024-07-24 02:13:50,822] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.32 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:50,822] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.07 | bwd_microstep: 146.93 | bwd_inner_microstep: 59.97 | bwd_allreduce_microstep: 86.93 | step_microstep: 57.97
[2024-07-24 02:13:50,823] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 146.92 | bwd_inner: 59.96 | bwd_allreduce: 86.93 | step: 57.98
[2024-07-24 02:13:51,077] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.70 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:51,077] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 146.50 | bwd_inner_microstep: 60.18 | bwd_allreduce_microstep: 86.30 | step_microstep: 57.48
[2024-07-24 02:13:51,078] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.49 | bwd_inner: 60.17 | bwd_allreduce: 86.30 | step: 57.49
[2024-07-24 02:13:51,331] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.83 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:51,332] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.10 | bwd_microstep: 146.33 | bwd_inner_microstep: 60.02 | bwd_allreduce_microstep: 86.29 | step_microstep: 57.91
[2024-07-24 02:13:51,332] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.10 | bwd: 146.32 | bwd_inner: 60.01 | bwd_allreduce: 86.29 | step: 57.92
[2024-07-24 02:13:51,585] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.64 | optimizer_gradients: 0.64 | optimizer_step: 0.92
[2024-07-24 02:13:51,586] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 146.21 | bwd_inner_microstep: 60.60 | bwd_allreduce_microstep: 85.58 | step_microstep: 57.79
[2024-07-24 02:13:51,586] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.25 | bwd: 146.20 | bwd_inner: 60.60 | bwd_allreduce: 85.58 | step: 57.81
[2024-07-24 02:13:51,839] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.74 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:51,839] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 145.97 | bwd_inner_microstep: 60.08 | bwd_allreduce_microstep: 85.86 | step_microstep: 57.37
[2024-07-24 02:13:51,839] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.96 | bwd_inner: 60.07 | bwd_allreduce: 85.86 | step: 57.38
[2024-07-24 02:13:52,092] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.75 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:52,093] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 145.48 | bwd_inner_microstep: 60.22 | bwd_allreduce_microstep: 85.22 | step_microstep: 58.42
[2024-07-24 02:13:52,093] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 145.47 | bwd_inner: 60.21 | bwd_allreduce: 85.22 | step: 58.43
[2024-07-24 02:13:52,345] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.58 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:52,345] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 145.54 | bwd_inner_microstep: 60.21 | bwd_allreduce_microstep: 85.31 | step_microstep: 57.11
[2024-07-24 02:13:52,345] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.53 | bwd_inner: 60.20 | bwd_allreduce: 85.31 | step: 57.12
Epoch: [0]  [  100/10009]  eta: 0:44:56  lr: 0.000001  min_lr: 0.000001  all_loss_mean: -0.2029 (-0.0598)  loss: -0.2007 (-0.0593)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2538  data: 0.0001  max mem: 9933
[2024-07-24 02:13:52,600] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.29 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:52,600] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.55 | bwd_inner_microstep: 59.87 | bwd_allreduce_microstep: 86.65 | step_microstep: 57.77
[2024-07-24 02:13:52,600] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.13 | bwd: 146.54 | bwd_inner: 59.86 | bwd_allreduce: 86.65 | step: 57.78
[2024-07-24 02:13:52,857] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.82 | optimizer_gradients: 0.60 | optimizer_step: 0.92
[2024-07-24 02:13:52,858] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 149.61 | bwd_inner_microstep: 60.25 | bwd_allreduce_microstep: 89.33 | step_microstep: 58.38
[2024-07-24 02:13:52,858] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 149.60 | bwd_inner: 60.24 | bwd_allreduce: 89.33 | step: 58.40
[2024-07-24 02:13:53,110] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.47 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:53,111] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 145.78 | bwd_inner_microstep: 59.75 | bwd_allreduce_microstep: 86.00 | step_microstep: 57.11
[2024-07-24 02:13:53,111] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 145.77 | bwd_inner: 59.74 | bwd_allreduce: 86.00 | step: 57.12
[2024-07-24 02:13:53,365] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.92 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:53,365] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.25 | bwd_microstep: 145.29 | bwd_inner_microstep: 59.60 | bwd_allreduce_microstep: 85.66 | step_microstep: 57.56
[2024-07-24 02:13:53,365] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.25 | bwd: 145.28 | bwd_inner: 59.59 | bwd_allreduce: 85.66 | step: 57.58
[2024-07-24 02:13:53,618] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.97 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:53,619] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 145.76 | bwd_inner_microstep: 60.08 | bwd_allreduce_microstep: 85.65 | step_microstep: 57.79
[2024-07-24 02:13:53,619] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 145.75 | bwd_inner: 60.06 | bwd_allreduce: 85.65 | step: 57.80
[2024-07-24 02:13:53,873] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.08 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:53,874] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 146.69 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 86.55 | step_microstep: 57.71
[2024-07-24 02:13:53,874] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 146.68 | bwd_inner: 60.10 | bwd_allreduce: 86.55 | step: 57.72
[2024-07-24 02:13:54,127] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.15 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:54,127] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 145.79 | bwd_inner_microstep: 60.06 | bwd_allreduce_microstep: 85.71 | step_microstep: 57.75
[2024-07-24 02:13:54,127] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.78 | bwd_inner: 60.05 | bwd_allreduce: 85.71 | step: 57.76
[2024-07-24 02:13:54,380] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.32 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:54,380] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.00 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 85.74 | step_microstep: 57.02
[2024-07-24 02:13:54,380] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 145.99 | bwd_inner: 60.22 | bwd_allreduce: 85.74 | step: 57.04
[2024-07-24 02:13:54,633] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.17 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:54,634] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 145.91 | bwd_inner_microstep: 59.80 | bwd_allreduce_microstep: 86.08 | step_microstep: 57.75
[2024-07-24 02:13:54,634] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.15 | bwd: 145.89 | bwd_inner: 59.79 | bwd_allreduce: 86.08 | step: 57.76
[2024-07-24 02:13:54,888] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.12 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:54,888] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 146.39 | bwd_inner_microstep: 60.72 | bwd_allreduce_microstep: 85.63 | step_microstep: 58.09
[2024-07-24 02:13:54,889] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.15 | bwd: 146.37 | bwd_inner: 60.71 | bwd_allreduce: 85.64 | step: 58.11
Epoch: [0]  [  110/10009]  eta: 0:44:38  lr: 0.000002  min_lr: 0.000002  all_loss_mean: -0.2419 (-0.0775)  loss: -0.2435 (-0.0773)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2539  data: 0.0001  max mem: 9933
[2024-07-24 02:13:55,142] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:55,142] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 145.63 | bwd_inner_microstep: 60.68 | bwd_allreduce_microstep: 84.90 | step_microstep: 57.49
[2024-07-24 02:13:55,142] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 145.61 | bwd_inner: 60.67 | bwd_allreduce: 84.90 | step: 57.50
[2024-07-24 02:13:55,395] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:55,396] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.21 | bwd_inner_microstep: 60.36 | bwd_allreduce_microstep: 85.81 | step_microstep: 57.52
[2024-07-24 02:13:55,396] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.20 | bwd_inner: 60.36 | bwd_allreduce: 85.81 | step: 57.53
[2024-07-24 02:13:55,649] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.21 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:55,650] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 146.08 | bwd_inner_microstep: 60.16 | bwd_allreduce_microstep: 85.88 | step_microstep: 57.83
[2024-07-24 02:13:55,650] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.15 | bwd: 146.06 | bwd_inner: 60.15 | bwd_allreduce: 85.89 | step: 57.84
[2024-07-24 02:13:55,904] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.94 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:55,905] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 146.42 | bwd_inner_microstep: 59.99 | bwd_allreduce_microstep: 86.40 | step_microstep: 58.36
[2024-07-24 02:13:55,905] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 146.41 | bwd_inner: 59.98 | bwd_allreduce: 86.40 | step: 58.37
[2024-07-24 02:13:56,159] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.38 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:56,160] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 145.68 | bwd_inner_microstep: 60.06 | bwd_allreduce_microstep: 85.59 | step_microstep: 58.10
[2024-07-24 02:13:56,160] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 145.67 | bwd_inner: 60.05 | bwd_allreduce: 85.59 | step: 58.11
[2024-07-24 02:13:56,415] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.89 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:56,416] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.53 | bwd_inner_microstep: 60.40 | bwd_allreduce_microstep: 86.10 | step_microstep: 58.20
[2024-07-24 02:13:56,416] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.51 | bwd_inner: 60.39 | bwd_allreduce: 86.10 | step: 58.22
[2024-07-24 02:13:56,669] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.01 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:56,670] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.48 | bwd_inner_microstep: 60.32 | bwd_allreduce_microstep: 86.13 | step_microstep: 57.67
[2024-07-24 02:13:56,670] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.46 | bwd_inner: 60.31 | bwd_allreduce: 86.13 | step: 57.68
[2024-07-24 02:13:56,929] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.96 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:13:56,929] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.68 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 86.71 | step_microstep: 62.76
[2024-07-24 02:13:56,930] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 146.67 | bwd_inner: 59.93 | bwd_allreduce: 86.72 | step: 62.77
[2024-07-24 02:13:57,182] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.99 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:57,183] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.17 | bwd_inner_microstep: 59.84 | bwd_allreduce_microstep: 86.29 | step_microstep: 57.51
[2024-07-24 02:13:57,183] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.14 | bwd_inner: 59.83 | bwd_allreduce: 86.29 | step: 57.52
[2024-07-24 02:13:57,435] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.62 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:57,435] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 145.75 | bwd_inner_microstep: 59.82 | bwd_allreduce_microstep: 85.90 | step_microstep: 57.22
[2024-07-24 02:13:57,435] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.74 | bwd_inner: 59.81 | bwd_allreduce: 85.90 | step: 57.24
Epoch: [0]  [  120/10009]  eta: 0:44:22  lr: 0.000002  min_lr: 0.000002  all_loss_mean: -0.2743 (-0.0951)  loss: -0.2703 (-0.0947)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2544  data: 0.0001  max mem: 9933
[2024-07-24 02:13:57,690] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.60 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:57,690] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.54 | bwd_inner_microstep: 60.32 | bwd_allreduce_microstep: 86.19 | step_microstep: 57.30
[2024-07-24 02:13:57,690] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.53 | bwd_inner: 60.31 | bwd_allreduce: 86.19 | step: 57.31
[2024-07-24 02:13:57,953] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.66 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:57,954] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 145.65 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 85.39 | step_microstep: 67.85
[2024-07-24 02:13:57,954] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 145.63 | bwd_inner: 60.22 | bwd_allreduce: 85.39 | step: 67.86
[2024-07-24 02:13:58,207] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.28 | optimizer_gradients: 0.61 | optimizer_step: 0.92
[2024-07-24 02:13:58,207] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 145.93 | bwd_inner_microstep: 60.00 | bwd_allreduce_microstep: 85.90 | step_microstep: 57.88
[2024-07-24 02:13:58,207] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 145.92 | bwd_inner: 59.99 | bwd_allreduce: 85.90 | step: 57.89
[2024-07-24 02:13:58,462] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.31 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:13:58,462] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.07 | bwd_microstep: 147.12 | bwd_inner_microstep: 59.73 | bwd_allreduce_microstep: 87.36 | step_microstep: 56.93
[2024-07-24 02:13:58,463] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 147.11 | bwd_inner: 59.72 | bwd_allreduce: 87.36 | step: 56.94
[2024-07-24 02:13:58,715] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.94 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:58,716] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.11 | bwd_microstep: 146.11 | bwd_inner_microstep: 60.19 | bwd_allreduce_microstep: 85.89 | step_microstep: 57.81
[2024-07-24 02:13:58,716] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.10 | bwd_inner: 60.18 | bwd_allreduce: 85.89 | step: 57.82
[2024-07-24 02:13:58,971] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.72 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:13:58,971] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 147.98 | bwd_inner_microstep: 60.36 | bwd_allreduce_microstep: 87.59 | step_microstep: 57.36
[2024-07-24 02:13:58,972] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 147.97 | bwd_inner: 60.35 | bwd_allreduce: 87.59 | step: 57.37
[2024-07-24 02:13:59,227] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.05 | optimizer_gradients: 0.63 | optimizer_step: 0.92
[2024-07-24 02:13:59,228] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 146.95 | bwd_inner_microstep: 60.38 | bwd_allreduce_microstep: 86.54 | step_microstep: 58.66
[2024-07-24 02:13:59,228] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.94 | bwd_inner: 60.37 | bwd_allreduce: 86.54 | step: 58.68
[2024-07-24 02:13:59,481] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.88 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:59,482] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.94 | bwd_inner_microstep: 60.40 | bwd_allreduce_microstep: 85.50 | step_microstep: 58.11
[2024-07-24 02:13:59,482] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.92 | bwd_inner: 60.39 | bwd_allreduce: 85.50 | step: 58.13
[2024-07-24 02:13:59,735] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.83 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:13:59,735] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.41 | bwd_inner_microstep: 60.15 | bwd_allreduce_microstep: 86.22 | step_microstep: 57.37
[2024-07-24 02:13:59,735] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.39 | bwd_inner: 60.15 | bwd_allreduce: 86.22 | step: 57.38
[2024-07-24 02:13:59,988] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.36 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:13:59,989] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 145.69 | bwd_inner_microstep: 60.27 | bwd_allreduce_microstep: 85.38 | step_microstep: 58.14
[2024-07-24 02:13:59,989] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 145.67 | bwd_inner: 60.26 | bwd_allreduce: 85.38 | step: 58.16
Epoch: [0]  [  130/10009]  eta: 0:44:09  lr: 0.000002  min_lr: 0.000002  all_loss_mean: -0.3063 (-0.1123)  loss: -0.3070 (-0.1119)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2549  data: 0.0002  max mem: 9933
[2024-07-24 02:14:00,243] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.14 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:14:00,244] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.36 | bwd_inner_microstep: 60.06 | bwd_allreduce_microstep: 86.27 | step_microstep: 58.26
[2024-07-24 02:14:00,244] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.35 | bwd_inner: 60.06 | bwd_allreduce: 86.27 | step: 58.28
[2024-07-24 02:14:00,497] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.97 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:00,498] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.11 | bwd_microstep: 146.66 | bwd_inner_microstep: 59.68 | bwd_allreduce_microstep: 86.96 | step_microstep: 57.62
[2024-07-24 02:14:00,498] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.65 | bwd_inner: 59.67 | bwd_allreduce: 86.96 | step: 57.63
[2024-07-24 02:14:00,752] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.56 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:00,753] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.11 | bwd_microstep: 146.87 | bwd_inner_microstep: 59.95 | bwd_allreduce_microstep: 86.89 | step_microstep: 58.41
[2024-07-24 02:14:00,753] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.86 | bwd_inner: 59.95 | bwd_allreduce: 86.89 | step: 58.42
[2024-07-24 02:14:01,006] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.91 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:14:01,007] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.10 | bwd_microstep: 146.36 | bwd_inner_microstep: 59.98 | bwd_allreduce_microstep: 86.35 | step_microstep: 57.97
[2024-07-24 02:14:01,007] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.11 | bwd: 146.35 | bwd_inner: 59.97 | bwd_allreduce: 86.35 | step: 57.99
[2024-07-24 02:14:01,262] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.31 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:01,263] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 33.16 | bwd_microstep: 146.52 | bwd_inner_microstep: 60.55 | bwd_allreduce_microstep: 85.93 | step_microstep: 56.80
[2024-07-24 02:14:01,263] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 33.17 | bwd: 146.50 | bwd_inner: 60.54 | bwd_allreduce: 85.93 | step: 56.81
[2024-07-24 02:14:01,515] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.60 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:01,516] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.30 | bwd_inner_microstep: 59.88 | bwd_allreduce_microstep: 86.39 | step_microstep: 57.12
[2024-07-24 02:14:01,516] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.29 | bwd_inner: 59.87 | bwd_allreduce: 86.39 | step: 57.13
[2024-07-24 02:14:01,769] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.41 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:01,769] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.88 | bwd_microstep: 145.96 | bwd_inner_microstep: 60.36 | bwd_allreduce_microstep: 85.57 | step_microstep: 56.94
[2024-07-24 02:14:01,770] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.89 | bwd: 145.95 | bwd_inner: 60.35 | bwd_allreduce: 85.57 | step: 56.96
[2024-07-24 02:14:02,022] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.91 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:02,023] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 145.85 | bwd_inner_microstep: 59.68 | bwd_allreduce_microstep: 86.14 | step_microstep: 57.81
[2024-07-24 02:14:02,023] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.84 | bwd_inner: 59.67 | bwd_allreduce: 86.14 | step: 57.83
[2024-07-24 02:14:02,276] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.84 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:02,277] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.15 | bwd_inner_microstep: 60.20 | bwd_allreduce_microstep: 85.93 | step_microstep: 57.33
[2024-07-24 02:14:02,277] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 146.14 | bwd_inner: 60.19 | bwd_allreduce: 85.93 | step: 57.35
[2024-07-24 02:14:02,531] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.13 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:14:02,532] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.20 | bwd_inner_microstep: 60.10 | bwd_allreduce_microstep: 86.07 | step_microstep: 57.97
[2024-07-24 02:14:02,532] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.19 | bwd_inner: 60.09 | bwd_allreduce: 86.07 | step: 57.98
Epoch: [0]  [  140/10009]  eta: 0:43:56  lr: 0.000002  min_lr: 0.000002  all_loss_mean: -0.3334 (-0.1288)  loss: -0.3381 (-0.1285)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2547  data: 0.0002  max mem: 9933
[2024-07-24 02:14:02,785] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.61 | optimizer_gradients: 0.61 | optimizer_step: 0.92
[2024-07-24 02:14:02,786] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 147.06 | bwd_inner_microstep: 59.85 | bwd_allreduce_microstep: 87.18 | step_microstep: 57.08
[2024-07-24 02:14:02,786] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.13 | bwd: 147.05 | bwd_inner: 59.85 | bwd_allreduce: 87.18 | step: 57.09
[2024-07-24 02:14:03,039] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.11 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:03,039] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.06 | bwd_inner_microstep: 59.92 | bwd_allreduce_microstep: 86.12 | step_microstep: 57.67
[2024-07-24 02:14:03,039] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.05 | bwd_inner: 59.91 | bwd_allreduce: 86.12 | step: 57.68
[2024-07-24 02:14:03,293] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.94 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:03,293] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 146.56 | bwd_inner_microstep: 59.90 | bwd_allreduce_microstep: 86.62 | step_microstep: 57.63
[2024-07-24 02:14:03,293] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.55 | bwd_inner: 59.90 | bwd_allreduce: 86.63 | step: 57.64
[2024-07-24 02:14:03,548] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.03 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:03,548] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.05 | bwd_microstep: 145.63 | bwd_inner_microstep: 59.81 | bwd_allreduce_microstep: 85.79 | step_microstep: 57.64
[2024-07-24 02:14:03,548] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.05 | bwd: 145.61 | bwd_inner: 59.80 | bwd_allreduce: 85.79 | step: 57.65
[2024-07-24 02:14:03,802] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.95 | optimizer_gradients: 0.62 | optimizer_step: 0.92
[2024-07-24 02:14:03,802] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 145.20 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 85.23 | step_microstep: 57.72
[2024-07-24 02:14:03,803] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.19 | bwd_inner: 59.93 | bwd_allreduce: 85.23 | step: 57.73
[2024-07-24 02:14:04,057] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.55 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:04,058] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.33 | bwd_inner_microstep: 59.75 | bwd_allreduce_microstep: 86.55 | step_microstep: 57.17
[2024-07-24 02:14:04,058] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.32 | bwd_inner: 59.74 | bwd_allreduce: 86.55 | step: 57.18
[2024-07-24 02:14:04,311] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.68 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:14:04,312] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 146.30 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 86.04 | step_microstep: 57.77
[2024-07-24 02:14:04,312] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 146.29 | bwd_inner: 60.22 | bwd_allreduce: 86.04 | step: 57.79
[2024-07-24 02:14:04,566] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.07 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:14:04,566] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.66 | bwd_inner_microstep: 60.22 | bwd_allreduce_microstep: 86.41 | step_microstep: 57.71
[2024-07-24 02:14:04,566] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.65 | bwd_inner: 60.21 | bwd_allreduce: 86.41 | step: 57.72
[2024-07-24 02:14:04,819] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.61 | optimizer_step: 0.92
[2024-07-24 02:14:04,819] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.07 | bwd_inner_microstep: 59.80 | bwd_allreduce_microstep: 86.24 | step_microstep: 57.60
[2024-07-24 02:14:04,819] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.05 | bwd_inner: 59.79 | bwd_allreduce: 86.24 | step: 57.61
[2024-07-24 02:14:05,074] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.28 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:05,074] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.01 | bwd_inner_microstep: 59.80 | bwd_allreduce_microstep: 86.18 | step_microstep: 57.08
[2024-07-24 02:14:05,074] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.00 | bwd_inner: 59.80 | bwd_allreduce: 86.18 | step: 57.09
Epoch: [0]  [  150/10009]  eta: 0:43:45  lr: 0.000002  min_lr: 0.000002  all_loss_mean: -0.3568 (-0.1444)  loss: -0.3569 (-0.1443)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2542  data: 0.0002  max mem: 9933
[2024-07-24 02:14:05,330] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.31 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:05,330] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.68 | bwd_inner_microstep: 60.39 | bwd_allreduce_microstep: 86.26 | step_microstep: 58.69
[2024-07-24 02:14:05,330] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 146.67 | bwd_inner: 60.38 | bwd_allreduce: 86.26 | step: 58.70
[2024-07-24 02:14:05,584] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.99 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:05,584] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.54 | bwd_inner_microstep: 60.26 | bwd_allreduce_microstep: 86.25 | step_microstep: 57.55
[2024-07-24 02:14:05,584] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.53 | bwd_inner: 60.25 | bwd_allreduce: 86.25 | step: 57.57
[2024-07-24 02:14:05,837] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.65 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:05,838] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.23 | bwd_inner_microstep: 59.88 | bwd_allreduce_microstep: 86.32 | step_microstep: 57.34
[2024-07-24 02:14:05,838] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.22 | bwd_inner: 59.88 | bwd_allreduce: 86.32 | step: 57.35
[2024-07-24 02:14:06,090] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.78 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:06,091] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.25 | bwd_microstep: 145.64 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 85.50 | step_microstep: 57.30
[2024-07-24 02:14:06,091] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.26 | bwd: 145.63 | bwd_inner: 60.10 | bwd_allreduce: 85.50 | step: 57.31
[2024-07-24 02:14:06,345] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.89 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:06,345] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.95 | bwd_microstep: 146.37 | bwd_inner_microstep: 60.02 | bwd_allreduce_microstep: 86.32 | step_microstep: 57.58
[2024-07-24 02:14:06,346] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.95 | bwd: 146.36 | bwd_inner: 60.01 | bwd_allreduce: 86.32 | step: 57.59
[2024-07-24 02:14:06,598] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.90 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:06,599] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.12 | bwd_inner_microstep: 60.18 | bwd_allreduce_microstep: 85.91 | step_microstep: 57.54
[2024-07-24 02:14:06,599] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.10 | bwd_inner: 60.17 | bwd_allreduce: 85.91 | step: 57.55
[2024-07-24 02:14:06,852] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.98 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:06,853] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.10 | bwd_inner_microstep: 60.07 | bwd_allreduce_microstep: 86.00 | step_microstep: 57.80
[2024-07-24 02:14:06,853] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 146.09 | bwd_inner: 60.07 | bwd_allreduce: 86.00 | step: 57.81
[2024-07-24 02:14:07,107] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.08 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:07,108] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.23 | bwd_inner_microstep: 60.26 | bwd_allreduce_microstep: 85.94 | step_microstep: 58.13
[2024-07-24 02:14:07,108] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.22 | bwd_inner: 60.26 | bwd_allreduce: 85.94 | step: 58.14
[2024-07-24 02:14:07,360] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.40 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:14:07,361] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.69 | bwd_inner_microstep: 59.76 | bwd_allreduce_microstep: 86.90 | step_microstep: 56.90
[2024-07-24 02:14:07,361] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.68 | bwd_inner: 59.76 | bwd_allreduce: 86.90 | step: 56.91
[2024-07-24 02:14:07,614] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.72 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:07,614] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.44 | bwd_inner_microstep: 59.84 | bwd_allreduce_microstep: 86.57 | step_microstep: 57.35
[2024-07-24 02:14:07,614] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.43 | bwd_inner: 59.84 | bwd_allreduce: 86.57 | step: 57.36
Epoch: [0]  [  160/10009]  eta: 0:43:35  lr: 0.000002  min_lr: 0.000002  all_loss_mean: -0.3772 (-0.1593)  loss: -0.3830 (-0.1595)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2540  data: 0.0001  max mem: 9933
[2024-07-24 02:14:07,867] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.17 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:07,867] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 146.33 | bwd_inner_microstep: 60.03 | bwd_allreduce_microstep: 86.27 | step_microstep: 56.81
[2024-07-24 02:14:07,868] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 146.32 | bwd_inner: 60.03 | bwd_allreduce: 86.27 | step: 56.82
[2024-07-24 02:14:08,120] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.68 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:08,121] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.03 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 85.77 | step_microstep: 57.24
[2024-07-24 02:14:08,121] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.02 | bwd_inner: 60.22 | bwd_allreduce: 85.77 | step: 57.26
[2024-07-24 02:14:08,374] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.03 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:08,374] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.26 | bwd_microstep: 145.94 | bwd_inner_microstep: 60.31 | bwd_allreduce_microstep: 85.59 | step_microstep: 57.65
[2024-07-24 02:14:08,374] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.26 | bwd: 145.93 | bwd_inner: 60.31 | bwd_allreduce: 85.59 | step: 57.66
[2024-07-24 02:14:08,631] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.91 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:08,632] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 32.02 | bwd_microstep: 148.59 | bwd_inner_microstep: 60.45 | bwd_allreduce_microstep: 88.10 | step_microstep: 57.65
[2024-07-24 02:14:08,632] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 32.03 | bwd: 148.57 | bwd_inner: 60.44 | bwd_allreduce: 88.10 | step: 57.66
[2024-07-24 02:14:08,886] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.09 | optimizer_gradients: 0.69 | optimizer_step: 0.92
[2024-07-24 02:14:08,887] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.86 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 85.89 | step_microstep: 58.51
[2024-07-24 02:14:08,887] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 145.85 | bwd_inner: 59.93 | bwd_allreduce: 85.89 | step: 58.53
[2024-07-24 02:14:09,140] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.09 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:09,140] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.34 | bwd_inner_microstep: 59.97 | bwd_allreduce_microstep: 86.33 | step_microstep: 57.60
[2024-07-24 02:14:09,140] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.33 | bwd_inner: 59.97 | bwd_allreduce: 86.33 | step: 57.62
[2024-07-24 02:14:09,393] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.43 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:14:09,393] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.25 | bwd_inner_microstep: 60.04 | bwd_allreduce_microstep: 86.17 | step_microstep: 57.10
[2024-07-24 02:14:09,393] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.23 | bwd_inner: 60.03 | bwd_allreduce: 86.17 | step: 57.11
[2024-07-24 02:14:09,646] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.65 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:09,646] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.11 | bwd_inner_microstep: 60.09 | bwd_allreduce_microstep: 85.99 | step_microstep: 57.15
[2024-07-24 02:14:09,646] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.10 | bwd_inner: 60.08 | bwd_allreduce: 85.99 | step: 57.16
[2024-07-24 02:14:09,900] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.80 | optimizer_gradients: 0.60 | optimizer_step: 0.92
[2024-07-24 02:14:09,901] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 146.52 | bwd_inner_microstep: 59.96 | bwd_allreduce_microstep: 86.52 | step_microstep: 58.14
[2024-07-24 02:14:09,901] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.25 | bwd: 146.50 | bwd_inner: 59.96 | bwd_allreduce: 86.52 | step: 58.15
[2024-07-24 02:14:10,154] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.38 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:10,155] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 146.11 | bwd_inner_microstep: 60.12 | bwd_allreduce_microstep: 85.96 | step_microstep: 57.91
[2024-07-24 02:14:10,155] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 146.10 | bwd_inner: 60.11 | bwd_allreduce: 85.96 | step: 57.92
Epoch: [0]  [  170/10009]  eta: 0:43:25  lr: 0.000003  min_lr: 0.000003  all_loss_mean: -0.3919 (-0.1734)  loss: -0.3945 (-0.1734)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2539  data: 0.0001  max mem: 9933
[2024-07-24 02:14:10,410] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.73 | optimizer_gradients: 0.61 | optimizer_step: 0.92
[2024-07-24 02:14:10,410] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.69 | bwd_inner_microstep: 60.13 | bwd_allreduce_microstep: 86.54 | step_microstep: 57.31
[2024-07-24 02:14:10,410] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.68 | bwd_inner: 60.12 | bwd_allreduce: 86.54 | step: 57.32
[2024-07-24 02:14:10,663] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.62 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:10,663] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.41 | bwd_inner_microstep: 60.42 | bwd_allreduce_microstep: 85.96 | step_microstep: 57.18
[2024-07-24 02:14:10,663] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.40 | bwd_inner: 60.41 | bwd_allreduce: 85.96 | step: 57.20
[2024-07-24 02:14:10,919] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.64 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:10,919] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.31 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 86.17 | step_microstep: 59.69
[2024-07-24 02:14:10,919] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.30 | bwd_inner: 60.10 | bwd_allreduce: 86.17 | step: 59.71
[2024-07-24 02:14:11,174] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.85 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:11,174] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.09 | bwd_microstep: 148.14 | bwd_inner_microstep: 59.74 | bwd_allreduce_microstep: 88.37 | step_microstep: 57.37
[2024-07-24 02:14:11,175] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.09 | bwd: 148.13 | bwd_inner: 59.73 | bwd_allreduce: 88.37 | step: 57.38
[2024-07-24 02:14:11,429] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.84 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:11,430] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 148.07 | bwd_inner_microstep: 60.23 | bwd_allreduce_microstep: 87.80 | step_microstep: 57.32
[2024-07-24 02:14:11,430] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 148.06 | bwd_inner: 60.22 | bwd_allreduce: 87.80 | step: 57.33
[2024-07-24 02:14:11,683] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.81 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:11,683] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.11 | bwd_microstep: 146.54 | bwd_inner_microstep: 60.03 | bwd_allreduce_microstep: 86.49 | step_microstep: 57.40
[2024-07-24 02:14:11,683] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.11 | bwd: 146.53 | bwd_inner: 60.02 | bwd_allreduce: 86.49 | step: 57.41
[2024-07-24 02:14:11,936] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.90 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:11,937] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.19 | bwd_inner_microstep: 59.95 | bwd_allreduce_microstep: 86.21 | step_microstep: 57.46
[2024-07-24 02:14:11,937] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.18 | bwd_inner: 59.94 | bwd_allreduce: 86.21 | step: 57.47
[2024-07-24 02:14:12,190] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:14:12,190] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.50 | bwd_inner_microstep: 60.08 | bwd_allreduce_microstep: 86.39 | step_microstep: 57.42
[2024-07-24 02:14:12,190] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.49 | bwd_inner: 60.07 | bwd_allreduce: 86.39 | step: 57.43
[2024-07-24 02:14:12,444] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.48 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:12,444] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 146.00 | bwd_inner_microstep: 60.37 | bwd_allreduce_microstep: 85.60 | step_microstep: 57.78
[2024-07-24 02:14:12,444] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 145.99 | bwd_inner: 60.36 | bwd_allreduce: 85.60 | step: 57.79
[2024-07-24 02:14:12,697] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.93 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:12,697] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.25 | bwd_microstep: 145.84 | bwd_inner_microstep: 60.15 | bwd_allreduce_microstep: 85.65 | step_microstep: 57.44
[2024-07-24 02:14:12,697] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.25 | bwd: 145.83 | bwd_inner: 60.15 | bwd_allreduce: 85.65 | step: 57.45
Epoch: [0]  [  180/10009]  eta: 0:43:17  lr: 0.000003  min_lr: 0.000003  all_loss_mean: -0.4069 (-0.1867)  loss: -0.4039 (-0.1867)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2541  data: 0.0001  max mem: 9933
[2024-07-24 02:14:12,951] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.86 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:12,951] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.61 | bwd_inner_microstep: 60.02 | bwd_allreduce_microstep: 86.56 | step_microstep: 57.44
[2024-07-24 02:14:12,951] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.59 | bwd_inner: 60.01 | bwd_allreduce: 86.56 | step: 57.46
[2024-07-24 02:14:13,205] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.98 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:13,205] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.26 | bwd_inner_microstep: 59.93 | bwd_allreduce_microstep: 86.30 | step_microstep: 57.59
[2024-07-24 02:14:13,205] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.25 | bwd_inner: 59.92 | bwd_allreduce: 86.30 | step: 57.60
[2024-07-24 02:14:13,461] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.41 | optimizer_gradients: 0.62 | optimizer_step: 0.91
[2024-07-24 02:14:13,461] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.62 | bwd_inner_microstep: 59.91 | bwd_allreduce_microstep: 86.68 | step_microstep: 60.19
[2024-07-24 02:14:13,461] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.61 | bwd_inner: 59.91 | bwd_allreduce: 86.68 | step: 60.21
[2024-07-24 02:14:13,730] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.86 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:13,730] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.07 | bwd_microstep: 162.17 | bwd_inner_microstep: 59.89 | bwd_allreduce_microstep: 102.25 | step_microstep: 57.39
[2024-07-24 02:14:13,730] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 162.16 | bwd_inner: 59.88 | bwd_allreduce: 102.25 | step: 57.40
[2024-07-24 02:14:13,983] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.60 | optimizer_gradients: 0.61 | optimizer_step: 0.91
[2024-07-24 02:14:13,983] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 145.85 | bwd_inner_microstep: 60.26 | bwd_allreduce_microstep: 85.56 | step_microstep: 57.23
[2024-07-24 02:14:13,984] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.84 | bwd_inner: 60.25 | bwd_allreduce: 85.56 | step: 57.24
[2024-07-24 02:14:14,237] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.92 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:14,237] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.43 | bwd_microstep: 146.28 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 86.30 | step_microstep: 57.39
[2024-07-24 02:14:14,238] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.43 | bwd: 146.26 | bwd_inner: 59.93 | bwd_allreduce: 86.30 | step: 57.40
[2024-07-24 02:14:14,490] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.23 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:14,491] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 146.48 | bwd_inner_microstep: 59.91 | bwd_allreduce_microstep: 86.53 | step_microstep: 56.97
[2024-07-24 02:14:14,491] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.47 | bwd_inner: 59.91 | bwd_allreduce: 86.53 | step: 56.98
[2024-07-24 02:14:14,743] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.98 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:14,744] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 145.59 | bwd_inner_microstep: 60.12 | bwd_allreduce_microstep: 85.44 | step_microstep: 57.52
[2024-07-24 02:14:14,744] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.58 | bwd_inner: 60.11 | bwd_allreduce: 85.44 | step: 57.53
[2024-07-24 02:14:14,997] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.99 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:14:14,998] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 146.21 | bwd_inner_microstep: 60.19 | bwd_allreduce_microstep: 85.98 | step_microstep: 57.99
[2024-07-24 02:14:14,998] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 146.19 | bwd_inner: 60.18 | bwd_allreduce: 85.98 | step: 58.00
[2024-07-24 02:14:15,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.72 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:14:15,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 145.44 | bwd_inner_microstep: 60.25 | bwd_allreduce_microstep: 85.16 | step_microstep: 57.49
[2024-07-24 02:14:15,251] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.43 | bwd_inner: 60.24 | bwd_allreduce: 85.16 | step: 57.50
Epoch: [0]  [  190/10009]  eta: 0:43:10  lr: 0.000003  min_lr: 0.000003  all_loss_mean: -0.4220 (-0.1991)  loss: -0.4196 (-0.1990)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2547  data: 0.0002  max mem: 9933
[2024-07-24 02:14:15,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.77 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:15,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.11 | bwd_microstep: 146.35 | bwd_inner_microstep: 59.78 | bwd_allreduce_microstep: 86.54 | step_microstep: 57.44
[2024-07-24 02:14:15,505] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.11 | bwd: 146.34 | bwd_inner: 59.78 | bwd_allreduce: 86.54 | step: 57.45
[2024-07-24 02:14:15,757] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.57 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:15,758] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 145.70 | bwd_inner_microstep: 60.38 | bwd_allreduce_microstep: 85.30 | step_microstep: 57.20
[2024-07-24 02:14:15,758] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.69 | bwd_inner: 60.37 | bwd_allreduce: 85.29 | step: 57.21
[2024-07-24 02:14:16,011] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.69 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:16,012] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 146.76 | bwd_inner_microstep: 60.22 | bwd_allreduce_microstep: 86.51 | step_microstep: 57.36
[2024-07-24 02:14:16,012] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 146.75 | bwd_inner: 60.21 | bwd_allreduce: 86.51 | step: 57.37
[2024-07-24 02:14:16,268] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.80 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:16,268] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.07 | bwd_microstep: 146.55 | bwd_inner_microstep: 59.70 | bwd_allreduce_microstep: 86.82 | step_microstep: 57.37
[2024-07-24 02:14:16,269] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.07 | bwd: 146.54 | bwd_inner: 59.69 | bwd_allreduce: 86.82 | step: 57.38
[2024-07-24 02:14:16,529] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.76 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:16,530] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 38.41 | bwd_microstep: 146.33 | bwd_inner_microstep: 60.26 | bwd_allreduce_microstep: 86.03 | step_microstep: 57.58
[2024-07-24 02:14:16,530] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 38.40 | bwd: 146.31 | bwd_inner: 60.25 | bwd_allreduce: 86.03 | step: 57.60
[2024-07-24 02:14:16,783] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.85 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:16,783] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.08 | bwd_inner_microstep: 59.93 | bwd_allreduce_microstep: 86.12 | step_microstep: 57.44
[2024-07-24 02:14:16,783] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.07 | bwd_inner: 59.92 | bwd_allreduce: 86.12 | step: 57.45
[2024-07-24 02:14:17,036] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.65 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:17,037] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 146.38 | bwd_inner_microstep: 60.32 | bwd_allreduce_microstep: 86.03 | step_microstep: 57.26
[2024-07-24 02:14:17,037] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 146.37 | bwd_inner: 60.31 | bwd_allreduce: 86.03 | step: 57.27
/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/PIL/TiffImagePlugin.py:900: UserWarning: Corrupt EXIF data.  Expecting to read 2 bytes but only got 0.
  warnings.warn(str(msg))
[2024-07-24 02:14:17,290] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.07 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:17,290] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.24 | bwd_microstep: 145.74 | bwd_inner_microstep: 60.09 | bwd_allreduce_microstep: 85.62 | step_microstep: 58.01
[2024-07-24 02:14:17,291] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.24 | bwd: 145.73 | bwd_inner: 60.08 | bwd_allreduce: 85.62 | step: 58.02
[2024-07-24 02:14:17,545] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.75 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:14:17,545] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.34 | bwd_microstep: 145.91 | bwd_inner_microstep: 60.20 | bwd_allreduce_microstep: 85.68 | step_microstep: 57.27
[2024-07-24 02:14:17,545] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.34 | bwd: 145.90 | bwd_inner: 60.19 | bwd_allreduce: 85.68 | step: 57.28
[2024-07-24 02:14:17,798] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.71 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:17,798] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 145.86 | bwd_inner_microstep: 59.86 | bwd_allreduce_microstep: 85.97 | step_microstep: 57.15
[2024-07-24 02:14:17,799] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.84 | bwd_inner: 59.85 | bwd_allreduce: 85.97 | step: 57.16
Epoch: [0]  [  200/10009]  eta: 0:43:03  lr: 0.000003  min_lr: 0.000003  all_loss_mean: -0.4291 (-0.2108)  loss: -0.4271 (-0.2105)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2550  data: 0.0002  max mem: 9933
[2024-07-24 02:14:18,053] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.28 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:18,053] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.11 | bwd_microstep: 146.56 | bwd_inner_microstep: 60.13 | bwd_allreduce_microstep: 86.40 | step_microstep: 57.89
[2024-07-24 02:14:18,053] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.55 | bwd_inner: 60.12 | bwd_allreduce: 86.40 | step: 57.90
[2024-07-24 02:14:18,306] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.01 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:18,307] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 146.29 | bwd_inner_microstep: 60.24 | bwd_allreduce_microstep: 86.02 | step_microstep: 57.65
[2024-07-24 02:14:18,307] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.28 | bwd_inner: 60.24 | bwd_allreduce: 86.02 | step: 57.66
[2024-07-24 02:14:18,560] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.89 | optimizer_gradients: 0.60 | optimizer_step: 0.92
[2024-07-24 02:14:18,561] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 146.33 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 86.18 | step_microstep: 57.73
[2024-07-24 02:14:18,561] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 146.32 | bwd_inner: 60.10 | bwd_allreduce: 86.18 | step: 57.74
[2024-07-24 02:14:18,817] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.35 | optimizer_gradients: 0.56 | optimizer_step: 0.92
[2024-07-24 02:14:18,817] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.11 | bwd_microstep: 146.30 | bwd_inner_microstep: 59.73 | bwd_allreduce_microstep: 86.54 | step_microstep: 59.91
[2024-07-24 02:14:18,817] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 146.29 | bwd_inner: 59.72 | bwd_allreduce: 86.54 | step: 59.92
[2024-07-24 02:14:19,079] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.53 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:19,080] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 40.08 | bwd_microstep: 146.27 | bwd_inner_microstep: 60.43 | bwd_allreduce_microstep: 85.80 | step_microstep: 57.49
[2024-07-24 02:14:19,080] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 40.07 | bwd: 146.26 | bwd_inner: 60.43 | bwd_allreduce: 85.81 | step: 57.50
[2024-07-24 02:14:19,333] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.36 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:19,334] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.83 | bwd_microstep: 146.43 | bwd_inner_microstep: 61.09 | bwd_allreduce_microstep: 85.31 | step_microstep: 56.97
[2024-07-24 02:14:19,334] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.83 | bwd: 146.42 | bwd_inner: 61.08 | bwd_allreduce: 85.31 | step: 56.98
[2024-07-24 02:14:19,588] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.69 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:19,588] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 147.39 | bwd_inner_microstep: 59.86 | bwd_allreduce_microstep: 87.51 | step_microstep: 57.35
[2024-07-24 02:14:19,588] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 147.38 | bwd_inner: 59.85 | bwd_allreduce: 87.51 | step: 57.36
[2024-07-24 02:14:19,841] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.70 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:19,841] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 145.80 | bwd_inner_microstep: 60.67 | bwd_allreduce_microstep: 85.10 | step_microstep: 57.17
[2024-07-24 02:14:19,841] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.79 | bwd_inner: 60.66 | bwd_allreduce: 85.10 | step: 57.19
[2024-07-24 02:14:20,095] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.65 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:20,096] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 146.55 | bwd_inner_microstep: 59.85 | bwd_allreduce_microstep: 86.67 | step_microstep: 57.25
[2024-07-24 02:14:20,096] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.15 | bwd: 146.54 | bwd_inner: 59.84 | bwd_allreduce: 86.67 | step: 57.26
[2024-07-24 02:14:20,347] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.75 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:20,348] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.15 | bwd_microstep: 145.21 | bwd_inner_microstep: 59.99 | bwd_allreduce_microstep: 85.19 | step_microstep: 57.28
[2024-07-24 02:14:20,348] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.15 | bwd: 145.20 | bwd_inner: 59.98 | bwd_allreduce: 85.19 | step: 57.29
Epoch: [0]  [  210/10009]  eta: 0:42:56  lr: 0.000003  min_lr: 0.000003  all_loss_mean: -0.4376 (-0.2217)  loss: -0.4365 (-0.2214)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2547  data: 0.0001  max mem: 9933
[2024-07-24 02:14:20,602] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.11 | optimizer_gradients: 0.59 | optimizer_step: 0.91
[2024-07-24 02:14:20,602] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.26 | bwd_microstep: 145.49 | bwd_inner_microstep: 60.12 | bwd_allreduce_microstep: 85.35 | step_microstep: 58.19
[2024-07-24 02:14:20,602] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.26 | bwd: 145.48 | bwd_inner: 60.11 | bwd_allreduce: 85.35 | step: 58.20
[2024-07-24 02:14:20,856] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.80 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:20,856] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.21 | bwd_microstep: 146.53 | bwd_inner_microstep: 60.25 | bwd_allreduce_microstep: 86.26 | step_microstep: 57.66
[2024-07-24 02:14:20,856] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.52 | bwd_inner: 60.24 | bwd_allreduce: 86.26 | step: 57.67
[2024-07-24 02:14:21,109] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.42 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:21,109] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.29 | bwd_microstep: 145.60 | bwd_inner_microstep: 60.07 | bwd_allreduce_microstep: 85.50 | step_microstep: 57.24
[2024-07-24 02:14:21,109] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.29 | bwd: 145.59 | bwd_inner: 60.06 | bwd_allreduce: 85.50 | step: 57.25
[2024-07-24 02:14:21,363] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:21,363] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.31 | bwd_microstep: 146.41 | bwd_inner_microstep: 60.21 | bwd_allreduce_microstep: 86.17 | step_microstep: 57.25
[2024-07-24 02:14:21,363] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.31 | bwd: 146.40 | bwd_inner: 60.21 | bwd_allreduce: 86.17 | step: 57.26
[2024-07-24 02:14:21,617] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.18 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:21,617] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 146.33 | bwd_inner_microstep: 59.53 | bwd_allreduce_microstep: 86.76 | step_microstep: 57.83
[2024-07-24 02:14:21,617] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 146.32 | bwd_inner: 59.52 | bwd_allreduce: 86.76 | step: 57.84
[2024-07-24 02:14:21,872] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.16 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:21,872] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.22 | bwd_microstep: 146.00 | bwd_inner_microstep: 59.92 | bwd_allreduce_microstep: 86.05 | step_microstep: 57.76
[2024-07-24 02:14:21,872] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.22 | bwd: 145.99 | bwd_inner: 59.92 | bwd_allreduce: 86.05 | step: 57.77
[2024-07-24 02:14:22,124] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.77 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:14:22,125] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.23 | bwd_inner_microstep: 59.54 | bwd_allreduce_microstep: 86.65 | step_microstep: 57.31
[2024-07-24 02:14:22,125] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 146.21 | bwd_inner: 59.54 | bwd_allreduce: 86.65 | step: 57.32
[2024-07-24 02:14:22,377] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.85 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:22,377] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.18 | bwd_microstep: 145.29 | bwd_inner_microstep: 60.19 | bwd_allreduce_microstep: 85.06 | step_microstep: 57.57
[2024-07-24 02:14:22,377] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 145.27 | bwd_inner: 60.19 | bwd_allreduce: 85.06 | step: 57.58
[2024-07-24 02:14:22,630] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.68 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:22,630] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.50 | bwd_inner_microstep: 59.89 | bwd_allreduce_microstep: 86.57 | step_microstep: 57.30
[2024-07-24 02:14:22,631] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.49 | bwd_inner: 59.89 | bwd_allreduce: 86.57 | step: 57.31
[2024-07-24 02:14:22,884] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.50 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:22,885] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 146.66 | bwd_inner_microstep: 59.76 | bwd_allreduce_microstep: 86.87 | step_microstep: 58.28
[2024-07-24 02:14:22,885] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.13 | bwd: 146.65 | bwd_inner: 59.75 | bwd_allreduce: 86.87 | step: 58.29
Epoch: [0]  [  220/10009]  eta: 0:42:49  lr: 0.000003  min_lr: 0.000003  all_loss_mean: -0.4454 (-0.2320)  loss: -0.4466 (-0.2318)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2542  data: 0.0001  max mem: 9933
[2024-07-24 02:14:23,138] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.58 | optimizer_gradients: 0.60 | optimizer_step: 0.92
[2024-07-24 02:14:23,139] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 145.60 | bwd_inner_microstep: 60.11 | bwd_allreduce_microstep: 85.46 | step_microstep: 57.65
[2024-07-24 02:14:23,139] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 145.59 | bwd_inner: 60.10 | bwd_allreduce: 85.46 | step: 57.67
[2024-07-24 02:14:23,394] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.27 | optimizer_gradients: 0.59 | optimizer_step: 0.92
[2024-07-24 02:14:23,394] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 33.10 | bwd_microstep: 146.13 | bwd_inner_microstep: 60.22 | bwd_allreduce_microstep: 85.88 | step_microstep: 56.93
[2024-07-24 02:14:23,394] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 33.09 | bwd: 146.12 | bwd_inner: 60.21 | bwd_allreduce: 85.88 | step: 56.94
[2024-07-24 02:14:23,647] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.79 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:23,648] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.00 | bwd_inner_microstep: 60.05 | bwd_allreduce_microstep: 85.92 | step_microstep: 57.74
[2024-07-24 02:14:23,648] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 145.99 | bwd_inner: 60.04 | bwd_allreduce: 85.92 | step: 57.76
[2024-07-24 02:14:23,903] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.53 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:23,903] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.35 | bwd_microstep: 147.73 | bwd_inner_microstep: 59.87 | bwd_allreduce_microstep: 87.83 | step_microstep: 57.33
[2024-07-24 02:14:23,903] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.35 | bwd: 147.72 | bwd_inner: 59.86 | bwd_allreduce: 87.83 | step: 57.34
[2024-07-24 02:14:24,157] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.06 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:24,158] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.97 | bwd_inner_microstep: 59.83 | bwd_allreduce_microstep: 87.12 | step_microstep: 57.70
[2024-07-24 02:14:24,158] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.96 | bwd_inner: 59.82 | bwd_allreduce: 87.12 | step: 57.72
[2024-07-24 02:14:24,411] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.09 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:24,412] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.31 | bwd_microstep: 145.83 | bwd_inner_microstep: 59.84 | bwd_allreduce_microstep: 85.96 | step_microstep: 58.08
[2024-07-24 02:14:24,412] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.32 | bwd: 145.82 | bwd_inner: 59.83 | bwd_allreduce: 85.96 | step: 58.09
[2024-07-24 02:14:24,665] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.89 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:14:24,665] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.13 | bwd_microstep: 146.55 | bwd_inner_microstep: 59.76 | bwd_allreduce_microstep: 86.76 | step_microstep: 57.34
[2024-07-24 02:14:24,665] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.13 | bwd: 146.54 | bwd_inner: 59.75 | bwd_allreduce: 86.76 | step: 57.35
[2024-07-24 02:14:24,920] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.05 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:24,920] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.12 | bwd_microstep: 147.23 | bwd_inner_microstep: 59.83 | bwd_allreduce_microstep: 87.37 | step_microstep: 57.87
[2024-07-24 02:14:24,920] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.12 | bwd: 147.22 | bwd_inner: 59.83 | bwd_allreduce: 87.37 | step: 57.88
[2024-07-24 02:14:25,175] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.02 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:25,175] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 146.61 | bwd_inner_microstep: 59.80 | bwd_allreduce_microstep: 86.78 | step_microstep: 57.98
[2024-07-24 02:14:25,176] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 146.60 | bwd_inner: 59.79 | bwd_allreduce: 86.78 | step: 58.00
[2024-07-24 02:14:25,430] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.00 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:25,431] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.25 | bwd_microstep: 146.54 | bwd_inner_microstep: 59.69 | bwd_allreduce_microstep: 86.82 | step_microstep: 58.70
[2024-07-24 02:14:25,431] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.25 | bwd: 146.53 | bwd_inner: 59.68 | bwd_allreduce: 86.82 | step: 58.71
Epoch: [0]  [  230/10009]  eta: 0:42:43  lr: 0.000003  min_lr: 0.000003  all_loss_mean: -0.4529 (-0.2417)  loss: -0.4559 (-0.2415)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2540  data: 0.0002  max mem: 9933
[2024-07-24 02:14:25,685] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.95 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:25,685] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.40 | bwd_microstep: 146.16 | bwd_inner_microstep: 60.53 | bwd_allreduce_microstep: 85.59 | step_microstep: 57.48
[2024-07-24 02:14:25,685] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.39 | bwd: 146.14 | bwd_inner: 60.52 | bwd_allreduce: 85.59 | step: 57.49
[2024-07-24 02:14:25,938] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.85 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:25,939] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.48 | bwd_inner_microstep: 60.09 | bwd_allreduce_microstep: 86.36 | step_microstep: 57.46
[2024-07-24 02:14:25,939] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.46 | bwd_inner: 60.08 | bwd_allreduce: 86.36 | step: 57.47
[2024-07-24 02:14:26,192] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.86 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:26,192] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 145.74 | bwd_inner_microstep: 60.04 | bwd_allreduce_microstep: 85.67 | step_microstep: 57.64
[2024-07-24 02:14:26,192] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 145.73 | bwd_inner: 60.04 | bwd_allreduce: 85.67 | step: 57.65
[2024-07-24 02:14:26,446] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.66 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:26,446] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.26 | bwd_microstep: 147.36 | bwd_inner_microstep: 59.88 | bwd_allreduce_microstep: 87.45 | step_microstep: 57.09
[2024-07-24 02:14:26,446] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.26 | bwd: 147.35 | bwd_inner: 59.87 | bwd_allreduce: 87.45 | step: 57.10
[2024-07-24 02:14:26,700] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.83 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:14:26,700] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.91 | bwd_inner_microstep: 60.15 | bwd_allreduce_microstep: 85.73 | step_microstep: 58.12
[2024-07-24 02:14:26,700] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.90 | bwd_inner: 60.14 | bwd_allreduce: 85.73 | step: 58.13
[2024-07-24 02:14:26,953] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.67 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:26,954] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.28 | bwd_microstep: 146.43 | bwd_inner_microstep: 60.45 | bwd_allreduce_microstep: 85.94 | step_microstep: 57.34
[2024-07-24 02:14:26,954] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.28 | bwd: 146.41 | bwd_inner: 60.45 | bwd_allreduce: 85.94 | step: 57.35
[2024-07-24 02:14:27,206] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.44 | optimizer_gradients: 0.60 | optimizer_step: 0.92
[2024-07-24 02:14:27,206] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.19 | bwd_microstep: 145.24 | bwd_inner_microstep: 60.17 | bwd_allreduce_microstep: 85.04 | step_microstep: 57.07
[2024-07-24 02:14:27,206] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.19 | bwd: 145.23 | bwd_inner: 60.16 | bwd_allreduce: 85.04 | step: 57.08
[2024-07-24 02:14:27,459] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.32 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:27,460] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 146.10 | bwd_inner_microstep: 59.86 | bwd_allreduce_microstep: 86.21 | step_microstep: 57.93
[2024-07-24 02:14:27,460] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 146.09 | bwd_inner: 59.85 | bwd_allreduce: 86.22 | step: 57.94
[2024-07-24 02:14:27,711] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.27 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:27,711] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.14 | bwd_microstep: 145.48 | bwd_inner_microstep: 59.94 | bwd_allreduce_microstep: 85.51 | step_microstep: 56.95
[2024-07-24 02:14:27,712] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.14 | bwd: 145.46 | bwd_inner: 59.93 | bwd_allreduce: 85.51 | step: 56.96
[2024-07-24 02:14:27,965] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.70 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:27,965] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 146.78 | bwd_inner_microstep: 60.08 | bwd_allreduce_microstep: 86.67 | step_microstep: 57.49
[2024-07-24 02:14:27,966] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.21 | bwd: 146.77 | bwd_inner: 60.07 | bwd_allreduce: 86.67 | step: 57.50
Epoch: [0]  [  240/10009]  eta: 0:42:37  lr: 0.000004  min_lr: 0.000004  all_loss_mean: -0.4613 (-0.2510)  loss: -0.4633 (-0.2510)  loss_scale: 65536.0000 (65536.0000)  weight_decay: 0.0500 (0.0500)  time: 0.2539  data: 0.0002  max mem: 9933
[2024-07-24 02:14:28,219] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.92 | optimizer_gradients: 0.57 | optimizer_step: 0.91
[2024-07-24 02:14:28,220] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 145.91 | bwd_inner_microstep: 60.05 | bwd_allreduce_microstep: 85.83 | step_microstep: 57.62
[2024-07-24 02:14:28,220] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 145.90 | bwd_inner: 60.04 | bwd_allreduce: 85.83 | step: 57.63
[2024-07-24 02:14:28,473] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 45.13 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:28,473] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.16 | bwd_microstep: 145.92 | bwd_inner_microstep: 59.51 | bwd_allreduce_microstep: 86.38 | step_microstep: 57.70
[2024-07-24 02:14:28,473] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.16 | bwd: 145.91 | bwd_inner: 59.50 | bwd_allreduce: 86.38 | step: 57.71
[2024-07-24 02:14:28,726] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.92 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:28,726] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.20 | bwd_microstep: 145.85 | bwd_inner_microstep: 59.88 | bwd_allreduce_microstep: 85.94 | step_microstep: 57.56
[2024-07-24 02:14:28,726] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.20 | bwd: 145.84 | bwd_inner: 59.88 | bwd_allreduce: 85.94 | step: 57.57
[2024-07-24 02:14:28,980] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.91 | optimizer_gradients: 0.57 | optimizer_step: 0.92
[2024-07-24 02:14:28,980] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.25 | bwd_microstep: 146.11 | bwd_inner_microstep: 60.05 | bwd_allreduce_microstep: 86.03 | step_microstep: 57.61
[2024-07-24 02:14:28,980] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.25 | bwd: 146.10 | bwd_inner: 60.04 | bwd_allreduce: 86.03 | step: 57.62
[2024-07-24 02:14:29,233] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.61 | optimizer_gradients: 0.58 | optimizer_step: 0.92
[2024-07-24 02:14:29,234] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 146.50 | bwd_inner_microstep: 60.01 | bwd_allreduce_microstep: 86.46 | step_microstep: 57.40
[2024-07-24 02:14:29,234] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.18 | bwd: 146.49 | bwd_inner: 60.00 | bwd_allreduce: 86.46 | step: 57.41
[2024-07-24 02:14:29,487] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.60 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:29,488] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.23 | bwd_microstep: 146.23 | bwd_inner_microstep: 59.95 | bwd_allreduce_microstep: 86.25 | step_microstep: 57.18
[2024-07-24 02:14:29,488] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.23 | bwd: 146.22 | bwd_inner: 59.94 | bwd_allreduce: 86.25 | step: 57.19
[2024-07-24 02:14:29,740] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | optimizer_allgather: 44.81 | optimizer_gradients: 0.58 | optimizer_step: 0.91
[2024-07-24 02:14:29,740] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd_microstep: 31.17 | bwd_microstep: 145.25 | bwd_inner_microstep: 59.99 | bwd_allreduce_microstep: 85.23 | step_microstep: 57.57
[2024-07-24 02:14:29,740] [INFO] [logging.py:96:log_dist] [Rank 0] time (ms) | fwd: 31.17 | bwd: 145.24 | bwd_inner: 59.98 | bwd_allreduce: 85.23 | step: 57.59
^CW0724 02:14:29.897000 140694527051584 torch/distributed/elastic/agent/server/api.py:741] Received Signals.SIGINT death signal, shutting down workers
W0724 02:14:29.897000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973435 closing signal SIGINT
W0724 02:14:29.897000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973436 closing signal SIGINT
W0724 02:14:29.897000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973437 closing signal SIGINT
W0724 02:14:29.898000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973438 closing signal SIGINT
^CW0724 02:14:30.044000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973435 closing signal SIGTERM
W0724 02:14:30.044000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973436 closing signal SIGTERM
W0724 02:14:30.044000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973437 closing signal SIGTERM
W0724 02:14:30.044000 140694527051584 torch/distributed/elastic/multiprocessing/api.py:851] Sending process 1973438 closing signal SIGTERM
^CTraceback (most recent call last):
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 733, in run
    result = self._invoke_run(role)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 876, in _invoke_run
    time.sleep(monitor_interval)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 76, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 1973359 got signal: 2

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 742, in run
    self._shutdown(e.sigval)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py", line 296, in _shutdown
    self._pcontext.close(death_sig)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 541, in close
    self._close(death_sig=death_sig, timeout=timeout)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 861, in _close
    handler.proc.wait(time_to_wait)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/subprocess.py", line 1209, in wait
    return self._wait(timeout=timeout)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/subprocess.py", line 1953, in _wait
    time.sleep(delay)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 76, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 1973359 got signal: 2

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/run.py", line 879, in main
    run(args)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/run.py", line 870, in run
    elastic_launch(
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 254, in launch_agent
    result = agent.run()
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 123, in wrapper
    result = f(*args, **kwargs)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 747, in run
    self._shutdown()
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py", line 296, in _shutdown
    self._pcontext.close(death_sig)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 541, in close
    self._close(death_sig=death_sig, timeout=timeout)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 861, in _close
    handler.proc.wait(time_to_wait)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/subprocess.py", line 1209, in wait
    return self._wait(timeout=timeout)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/subprocess.py", line 1953, in _wait
    time.sleep(delay)
  File "/home/xiaofeng.wu/anaconda3/envs/itpn-py310/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 76, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 1973359 got signal: 2
^C(itpn-py310) (base) xiaofeng.wu@Fairfax4way04RTX4090:/home/xiaofeng.wu/prjs/iTPN/CLIP_as_supervision$
```