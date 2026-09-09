I k![DINOv2 SALAD](assets/dino_salad_title.png)
# Optimal Transport Aggregation for Visual Place Recognition
Sergio Izquierdo, Javier Civera

Code and models for Optimal Transport Aggregation for Visual Place Recognition (DINOv2 SALAD).

## Summary

We introduce DINOv2 SALAD, a Visual Place Recognition model that achieves state-of-the-art results on common benchmarks. We introduce two main contributions:
 - Using a finetuned DINOv2 encoder to get richer and more powerful features.
 - A new aggregation technique based on optimal transport to create a global descriptor based on optimal transport. This aggregation extends NetVLAD to consider feature-to-cluster relations as well as cluster-to-features. Besides, it includes a dustbin to discard uninformative features.

For more details, check the paper at [arXiv](https://arxiv.org/abs/2311.15937).

![Method](assets/method.jpg)

## Setup

For the standalone uv environment, run from this directory:

```bash
uv sync --locked
uv run python train_megaloc.py --help
uv run python -m unittest discover -s tests -v
```

This creates `.venv` with Python 3.12, PyTorch 2.8 / CUDA 12.8, matching
torchvision and xFormers, Lightning, metric-learning, Parquet, and evaluation
dependencies. FAISS retrieval uses the CPU (`faiss_gpu=False`); model training
uses CUDA. `uv.lock` records the installed versions. This environment is separate
from `localizer-service/.venv`.

The original code was tested on PyTorch 2.1.0 with CUDA 12.1 and Xformers. Its
Conda environment remains available:
```bash
conda env create -f environment.yml
```

To quickly test and use our model, you can use Torch Hub:
```python
import torch
model = torch.hub.load("serizba/salad", "dinov2_salad")
model.eval()
model.cuda()
```

## Dataset

For training, download [GSV-Cities](https://github.com/amaralibey/gsv-cities) dataset. For evaluation download the desired datasets ([MSLS](https://github.com/FrederikWarburg/mapillary_sls), [NordLand](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W), [SPED](https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W), or [Pittsburgh](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/))

## Train

Select preprocessed MegaLoc schedules with a data manifest:

```yaml
path: ..
data:
  - name: v1_d4_vanilla
    subsets:
      - gsv_cities
      - megascenes
      - msls
      - sf_xl_frontal
      - sf_xl_lateral
```

`path` is relative to the manifest. Each `name` identifies a standalone dataset
folder containing `batches/`, `meta/`, and its image paths. Only listed subsets
are loaded, and their YAML order is preserved. Remove a dataset or subset from
the manifest to disable it.

```python
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataloaders.MegaLocDataset import MegaLocDataset

transform = T.Compose([
    T.Resize((224, 224)),
    T.RandAugment(num_ops=3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = MegaLocDataset("/path/to/data.yaml", transform)
loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4)
for iteration in loader:
    for subset, (images, labels) in iteration.items():
        # images: [128, 3, 224, 224]; labels: [128], local to this subset.
        pass  # Compute each subset's loss separately; step once per iteration.
```

The selected schedules must have the same length (4 for the snapshot, 40,000
for the full dataset). Metadata paths are resolved from their standalone dataset
folder, so they may reference shared images with relative paths.

`train_megaloc.py` trains DINOv3 with MegaLoc's head: SALAD with 64 clusters,
256 channels per cluster, and a 256-dimensional global token, followed by a
learned 16,640 → 8,448 linear projection and L2 normalization. The last four
backbone blocks and the entire head are trained using gradient caching.
Each iteration sums all discovered subset losses before one optimizer update.
Each launch creates `logs/YYYY-MM-DD_HH-MM-SS_microseconds/` containing resolved
`config.yaml` and `data.yaml` files, a `metrics.csv` row for every iteration,
and full checkpoints under `checkpoints/`. Periodic checkpoints are retained every 5,000
iterations by default, while `checkpoints/last.ckpt` always records the final
state. Model, optimization, augmentation, iteration, GradCache, seed, and
checkpoint settings live in `configs/train_megaloc.yaml`; `iterations: null`
consumes the complete schedule. Training requires a CUDA GPU. Launch from this
directory with:

```bash
uv run python train_megaloc.py \
    --data /mnt/e/D_WORK/Localizer/data/datasets/processed/MegaLoc/train/configs/v1_d_vanilla.yaml \
    --config configs/train_megaloc.yaml \
    --num-workers 0
```

Validation is disabled in this entrypoint. The projection is optional in SALAD
(`agg_config["output_dim"]`); existing configurations keep their original head.

Training is done on GSV-Cities for 4 complete epochs. It requires around 30 minutes on an NVIDIA RTX 3090. For training DINOv2 SALAD run:
```bash
python3 main.py
```

After training, logs and checkpoints should be on the `logs` dir.

## Evaluation

You can download a pretrained DINOv2 SALAD model from here:
<table>
 <thead>
   <tr>
    <th>Model Name</th>
    <th>Descriptor size</th>
    <th>Download link</th>
   </tr>
 </thead>
 <tbody>
   <tr>
     <td>dino_salad</td>
     <td>8192+256</td>
     <td> <a href="https://drive.google.com/file/d/1u83Dmqmm1-uikOPr58IIhfIzDYwFxCy1/view?usp=sharing">download</a></td>
   </tr>
   <tr>
     <td>dino_salad_512_32</td>
     <td>512 + 32</td>
     <td> <a href="https://drive.google.com/file/d/18SljgYj0mErBvuMoVYSJpI9BIQ7aDWDB/view?usp=sharing">download</a></td>
   </tr>
   <tr>
     <td>dino_salad_2048_64</td>
     <td>2048+64</td>
     <td> <a href="https://drive.google.com/file/d/1g0T5kCHfV6T-V1GWA1BlVGZb2KWzGIty/view?usp=sharing">download</a></td>
   </tr>
 </tbody>
</table>

For evaluating run:

```bash
python3 eval.py --ckpt_path 'weights/dino_salad.ckpt' --image_size 322 322 --batch_size 256 --val_datasets MSLS Nordland
```

<table>
<thead>
  <tr>
    <th colspan="3">MSLS Challenge</th>
    <th colspan="3">MSLS Val</th>
    <th colspan="3">NordLand</th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>75.0</td>
    <td>88.8</td>
    <td>91.3</td>
    <td>92.2</td>
    <td>96.4</td>
    <td>97.0</td>
    <td>76.0</td>
    <td>89.2</td>
    <td>92.0</td>
  </tr>
</tbody>
</table>

## Acknowledgements
This code is based on the amazing work of:
 - [MixVPR](https://github.com/amaralibey/MixVPR)
 - [GSV-Cities](https://github.com/amaralibey/gsv-cities)
 - [DINOv2](https://github.com/facebookresearch/dinov2)

## Cite
Here is the bibtex to cite our paper
```
@InProceedings{Izquierdo_CVPR_2024_SALAD,
    author    = {Izquierdo, Sergio and Civera, Javier},
    title     = {Optimal Transport Aggregation for Visual Place Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
}
```
