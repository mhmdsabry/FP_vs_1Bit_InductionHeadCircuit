# FP_vs_1Bit_InductionHeadCircuit
Comparing the full-precision Query, Keys, and Values matrices with their 1-bit counterparts in a two-layer, attention-only transformer trained on a synthetic copying task.

## Induction Heads and Induction Scores:
-
## Experiments:
We substituted the Query, Key, and Value linear layers with BitLinear layers, as outlined in [Hongyu Wang et al., 2023's BitNet](https://arxiv.org/pdf/2310.11453.pdf). Our aim was to understand the behavioural differences of induction heads in BitLinear layers in contrast to those in full-precision (FP) linear layers, across various configurations. To this end, we constructed a range of induction head circuits within two-layer, attention-only transformers, utilising every possible combination of the following:

- embed_sizes=(64 128 256 512 768 1024 2048 8192)
- num_heads=(2 4 8 16)
- norms=('pre_norm' 'post_norm' 'no_norm')
- positions_encodings=('learnable' 'trigonometric' 'no_pos')


## code:

#### Environment:
```
conda env create -f environment.yml
conda activate ihc_env
```
#### Train an induction circuit
config.ini contains models and training configs, feel free to tweak, then run:
```
python train.py -c config.ini --embed_size "$embed_size" --num_head "$head" --pos_embed "$pos_embed" --norm_mode "$norm" --seed "$seed"
```

