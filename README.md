# nafnet-jax

A Jax / Flax impl of the paper "Simple Baselines for Image Restoration"

This impl is heavyily based on the GitHub repo of [vision-transformer](https://github.com/google-research/vision_transformer) 
\& the original PyTorch impl [NAFNet](https://github.com/megvii-research/NAFNet).


## Usage

```
python -m nafnet.main --workdir=./logs \
    --config=$(pwd)/nafnet/configs/config_nafnet.py:w32 \
    --config.base_lr=0.01
```

## Citations

```
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```

## Acknowledgements

This work is supported with Cloud TPUs from Google's TPU Research Cloud (TRC).

