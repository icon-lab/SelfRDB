import os
import glob
import shutil
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
import nibabel as nib


def split_data(data_root, output_root):
    """
    Split data into train/val/test sets.

    Expected original data folder structure:
    <data_root>/
     ├── BraTS2021_00000/
     │   ├── BraTS2021_00000_flair.nii.gz
     │   ├── BraTS2021_00000_t1.nii.gz
     │   ├── BraTS2021_00000_t1ce.nii.gz
     │   └── BraTS2021_00000_t2.nii.gz
     ├── BraTS2021_00001/
     └── ...
    """
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")
    test_dir = os.path.join(output_root, "test")

    all_patients = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("BraTS")]
    all_patients = sorted(all_patients, key=lambda x: int(x.split("_")[-1]))
    random.shuffle(all_patients)

    train_patients = all_patients[:25]
    val_patients = all_patients[25:30]
    test_patients = all_patients[30:40]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Saving train/val/test splits")

    for p in train_patients:
        shutil.copytree(os.path.join(data_root, p), os.path.join(train_dir, p), dirs_exist_ok=True)
    for p in val_patients:
        shutil.copytree(os.path.join(data_root, p), os.path.join(val_dir, p), dirs_exist_ok=True)
    for p in test_patients:
        shutil.copytree(os.path.join(data_root, p), os.path.join(test_dir, p), dirs_exist_ok=True)


def create_brats_dataset(
    brats_root,
    output_root,
    slice_range=(27, 127),
    mask_threshold=0.1,
    modalities=None
):
    """
    Create a dataset from BRATS in the following format:
    <dataset>/
     ├── <modality_a>/
     │   ├── train/
     │   ├── val/
     │   └── test/
     ├── <modality_b>/
     │   ├── train/
     │   ├── val/
     │   └── test/
     ...

    Expected original BRATS dataset folder structure:
    <brats_root>/
     ├── train/
     │   ├── BraTS2021_00000/
     │   │   ├── BraTS2021_00000_flair.nii.gz
     │   │   ├── BraTS2021_00000_t1.nii.gz
     │   │   ├── BraTS2021_00000_t1ce.nii.gz
     │   │   └── BraTS2021_00000_t2.nii.gz
     │   └── BraTS2021_00001/
     │       └── ...
     ├── val/
     └── test/
    """
    if modalities is None:
        modalities = ["t1", "t2", "flair"]

    # Example splitting logic: 
    # We assume you have some lists or a method to define train/val/test
    dataset_splits = {
        "train": glob.glob(os.path.join(brats_root, "train", "*/")),
        "val": glob.glob(os.path.join(brats_root, "val", "*/")),
        "test": glob.glob(os.path.join(brats_root, "test", "*/"))
    }

    for modality in modalities:
        modality_dir = Path(output_root) / modality
        for split_type, list_of_cases in dataset_splits.items():
            split_dir = modality_dir / split_type
            split_dir.mkdir(parents=True, exist_ok=True)

            slice_idx = 0
            for case_path in tqdm(list_of_cases, desc=f"Processing {modality}->{split_type}"):
                # Path to a NIfTI file for the chosen modality
                nifti_file = os.path.join(case_path, f"{os.path.basename(case_path.strip('/'))}_{modality}.nii.gz")
                if not os.path.isfile(nifti_file):
                    continue

                img = nib.load(nifti_file).get_fdata()

                # Convert to slices
                for i in tqdm(range(*slice_range), desc="Slices", leave=False):
                    slice_data = img[:, :, i]
                    
                    # Rotate 90 deg clockwise
                    slice_data = np.rot90(slice_data, -1)
                    
                    # Scale to [0,1]
                    slice_data = slice_data - slice_data.min()
                    slice_data = slice_data / (slice_data.max() + 1e-8)

                    slice_filename = split_dir / f"slice_{slice_idx}.npy"
                    
                    np.save(str(slice_filename), slice_data)
                    slice_idx += 1
                    

    # Create masks from T1 for test split
    mask_dir = Path(output_root) / "mask" / "test"
    mask_dir.mkdir(parents=True, exist_ok=True)

    test_cases = dataset_splits["test"]
    slice_idx = 0
    for case_path in tqdm(test_cases, desc="Creating T1 masks"):
        t1_file = os.path.join(case_path, f"{os.path.basename(case_path.strip('/'))}_t1.nii.gz")
        if not os.path.isfile(t1_file):
            continue
        image = nib.load(t1_file).get_fdata()
        for i in tqdm(range(*slice_range), desc="Slices", leave=False):
            slice_data = image[..., i]
            slice_data = np.rot90(slice_data, -1)
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() + 1e-8)
            mask = (slice_data > mask_threshold).astype(np.uint8)
            np.save(mask_dir / f"slice_{slice_idx}.npy", mask)
            slice_idx += 1
    

def main():
    # Example usage
    brats_root = "datasets/BRATS2021/raw"
    output_root = "datasets/BRATS2021/"
    slice_range = (27, 127)
    mask_threshold = 0.1
    modalities = ["t1", "t2", "flair"]

    # Create train/val/test splits
    split_data(brats_root, brats_root)

    # Create dataset
    create_brats_dataset(
        brats_root=brats_root,
        output_root=output_root,
        slice_range=slice_range,
        mask_threshold=mask_threshold,
        modalities=modalities
    )

if __name__ == "__main__":
    main()
