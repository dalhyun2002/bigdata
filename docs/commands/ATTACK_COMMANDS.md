# Attack commands

## Baseline EfficientNet-B0

Existing baseline attack results are stored in:

```text
attacks/Adversarial_Attacks/baseline_efficientnetb0_seed4/
```

To reproduce the baseline attack:

```powershell
python attacks\adversarial_pgd_layercam.py --model-name baseline_efficientnetb0 --seed 4 --methods standard_pgd,layercam_masked_pgd --epsilons 0.125,0.25,0.5,1 --steps 10 --batch-size 16 --num-workers 4
```

## Proposed EfficientNet-B0 + SE + MLP Head

Run the proposed model under the same attack settings:

```powershell
python attacks\adversarial_pgd_layercam.py --model-name proposed_efficientnetb0_se_mlp512_full --seed 2 --checkpoint results\efficientnet_grid\weights\best_grid_efficientnetb0_se_mlp512_full_pretrained_seed2.pth --methods standard_pgd,layercam_masked_pgd --epsilons 0.125,0.25,0.5,1 --steps 10 --batch-size 16 --num-workers 4
```

The proposed attack results are saved to:

```text
attacks/Adversarial_Attacks/proposed_efficientnetb0_se_mlp512_full_seed2/
```

## Merge attack summaries

After the proposed attack finishes, merge the baseline and proposed results:

```powershell
python attacks\compare_attack_results.py
```

The merged comparison table is saved to:

```text
attacks/Adversarial_Attacks/attack_comparison_summary.csv
```

## GPU example

On RunPod or another CUDA machine, use AMP and a larger batch size if memory allows:

```bash
python attacks/adversarial_pgd_layercam.py --model-name proposed_efficientnetb0_se_mlp512_full --seed 2 --checkpoint results/efficientnet_grid/weights/best_grid_efficientnetb0_se_mlp512_full_pretrained_seed2.pth --methods standard_pgd,layercam_masked_pgd --epsilons 0.125,0.25,0.5,1 --steps 10 --batch-size 32 --num-workers 8 --amp
```
