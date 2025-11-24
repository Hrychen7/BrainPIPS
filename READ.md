# BrainPIPS: Progressive Self-Supervised Learning with Individualized Parcellation

This document provides instructions on how to run the BrainPIPS framework, which consists of three main stages:
1.  **Pre-training**: Learn robust representations of brain networks using our progressive self-supervised learning method.
2.  **Deviation Calculation**: Extract the individualized subnetwork deviation patterns from the pre-trained model.
3.  **Fine-tuning**: Fine-tune the pre-trained model on a downstream classification task, incorporating the calculated deviation information.

---

## Step 1: Pre-training

This step pre-trains the model on your entire dataset using a self-supervised objective. The command below runs the pre-training for a single fold. You should typically run this for all folds (e.g., in a loop from `fold0` to `fold9`).

```bash
python main_pretrain.py \
    --root /path/to/your/dataset/root \
    --csv /path/to/your/dataset_split.csv \
    --rho_0 0.5 \
    --model mae_BNTF_base \
    --output_dir ./output/pretrain_checkpoints \
    --log_dir ./output/pretrain_logs \
    --fold 0
```

#### Argument Explanation:
- `--root`: The root directory where your subject data (e.g., `.npy` files) is stored.
- `--csv`: Path to the CSV file that defines the dataset splits for training, validation, and testing for each fold.
- `--rho_0`: The initial transport mass for the progressive learning curriculum. This is a key hyperparameter.
- `--model`: The name of the model architecture to use (e.g., `mae_BNTF_base`).
- `--output_dir`: Directory where the pre-trained model checkpoints will be saved.
- `--log_dir`: Directory where the training logs will be stored.
- `--fold`: The specific cross-validation fold to run (e.g., 0).

> **Note**: After this step, you will have a pre-trained checkpoint for each fold, typically named `checkpoint-XXX.pth`, where `XXX` is the final epoch number.

---

## Step 2: Deviation Calculation

Using the pre-trained model from Step 1, this script calculates the subnetwork deviation matrix for each subject in the dataset.

```bash
python deviated.py \
    --root /path/to/your/dataset/root \
    --csv /path/to/your/dataset_split.csv \
    --checkpoint ./output/pretrain_checkpoints/YOUR_EXPERIMENT_NAME/fold0/checkpoint-XXX.pth \
    --save_dir ./output/deviation_data
```

#### Argument Explanation:
- `--root` & `--csv`: Same as in the pre-training step.
- `--checkpoint`: **Crucially**, provide the full path to the pre-trained model checkpoint generated in Step 1. Remember to replace `YOUR_EXPERIMENT_NAME` and `checkpoint-XXX.pth` with your actual directory and file names.
- `--save_dir`: The root directory where the calculated deviation matrices will be saved. The script will automatically create a `Flow` subdirectory inside it.

> **Note**: This step will generate a set of `.npy` files in the `./output/deviation_data/Flow/` directory, with each file corresponding to a subject's deviation matrix.

---

## Step 3: Fine-tuning for Downstream Classification

Finally, this step fine-tunes the pre-trained model on a specific classification task, leveraging both the learned representations and the deviation matrices.

```bash
python finetune.py \
    --root /path/to/your/dataset/root \
    --csv /path/to/your/dataset_split.csv \
    --deviated_root ./output/deviation_data/Flow \
    --resume ./output/pretrain_checkpoints/YOUR_EXPERIMENT_NAME/fold0/checkpoint-XXX.pth \
    --output_dir ./output/finetune_results \
    --fold 0 \
    --batch_size 64
```

#### Argument Explanation:
- `--root` & `--csv`: Same as before.
- `--deviated_root`: Path to the directory containing the deviation `.npy` files generated in Step 2.
- `--resume`: The path to the pre-trained model checkpoint from Step 1. This is used to initialize the model weights for fine-tuning.
- `--output_dir`: Directory where the fine-tuned models and classification logs will be saved.
- `--fold`: The specific cross-validation fold to run, which should correspond to the fold used for the pre-trained checkpoint.
- `--batch_size`: The batch size for the fine-tuning process.


By following these three steps, you can fully replicate the BrainPIPS training and evaluation pipeline. Remember to adapt the paths and hyperparameters according to your specific dataset and experimental setup.
