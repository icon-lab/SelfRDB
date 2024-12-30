import os
import numpy as np
import nibabel as nib
import glob
from pathlib import Path
from tqdm import tqdm


def create_brats_dataset(brats_root, output_root, modalities=None):
    """
    Create a dataset from BRATS in the format:
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
        modalities = ["t1", "t2", "t1ce", "flair"]

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

            for case_path in tqdm(list_of_cases, desc=f"Processing {modality}->{split_type}"):
                # Path to a NIfTI file for the chosen modality
                nifti_file = os.path.join(case_path, f"{os.path.basename(case_path.strip('/'))}_{modality}.nii.gz")
                if not os.path.isfile(nifti_file):
                    continue

                img = nib.load(nifti_file).get_fdata()
                # Convert to slices
                for slice_idx in tqdm(range(img.shape[2]), desc="Slices", leave=False):
                    slice_data = img[:, :, slice_idx]
                    
                    # Rotate 90 deg clockwise
                    slice_data = np.rot90(slice_data, -1)
                    
                    # Scale to [0,1]
                    slice_data = slice_data - slice_data.min()
                    slice_data = slice_data / (slice_data.max() + 1e-8)

                    slice_filename = split_dir / f"slice_{slice_idx}.npy"
                    np.save(str(slice_filename), slice_data)

def main():
    # Example usage
    brats_root = "/path/to/brats"
    output_root = "/path/to/output/dataset"
    modalities = ["t1", "t2", "flair"]
    create_brats_dataset(brats_root, output_root, modalities)

if __name__ == "__main__":
    main()
