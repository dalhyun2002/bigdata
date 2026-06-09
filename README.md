# MAR20 Aircraft Classification Robustness

This repository contains the code and compact artifacts for a MAR20 aircraft classification robustness experiment.

## Included

- Training and evaluation scripts
- Attack scripts for Standard PGD and LayerCAM Masked PGD
- Final comparison summaries
- Three selected model checkpoints:
  - `baseline_efficientnetb0`
  - `proposed_efficientnetb0_se_mlp512_full`
  - `scratch_small_maxpool_silu_pool4mlp`
- Small raw/cropped image samples under `MAR20/sample/`
- Matching sample Horizontal Bounding Box XML annotations under `MAR20/sample/Annotations/Horizontal Bounding Boxes/`

## Not Included

The full MAR20 dataset, full cropped classification dataset, all experiment checkpoints, and bulk generated images are intentionally excluded to keep the repository small.

## Main Commands

Prepare the full dataset locally after placing MAR20 source files:

```powershell
python prepare_dataset.py
```

Run baseline attack:

```powershell
python attacks\adversarial_pgd_layercam.py --model-name baseline_efficientnetb0 --seed 4 --methods standard_pgd,layercam_masked_pgd --epsilons 0.125,0.25,0.5,1 --steps 10 --batch-size 16 --num-workers 4
```

Run proposed model attack:

```powershell
python attacks\adversarial_pgd_layercam.py --model-name proposed_efficientnetb0_se_mlp512_full --seed 2 --checkpoint results\efficientnet_grid\weights\best_grid_efficientnetb0_se_mlp512_full_pretrained_seed2.pth --methods standard_pgd,layercam_masked_pgd --epsilons 0.125,0.25,0.5,1 --steps 10 --batch-size 16 --num-workers 4
```

Merge attack summaries:

```powershell
python attacks\compare_attack_results.py
```
