from abc import abstractmethod
import os
import yaml
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L


class BaseDataset(Dataset):
    def __init__(self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True
    ):
        self.data_dir = data_dir
        self.target_modality= target_modality
        self.source_modality = source_modality
        self.stage = stage
        self.image_size = image_size
        self.norm = norm
        self.padding = padding
        self.original_shape = None

    @abstractmethod
    def _load_data(self, contrast):
        pass

    def _pad_data(self, data):
        """ Pad data to image_size x image_size """
        H, W = data.shape[-2:]

        pad_top = (self.image_size - H) // 2
        pad_bottom = self.image_size - H - pad_top
        pad_left = (self.image_size - W) // 2
        pad_right = self.image_size - W - pad_left

        return np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))

    def _normalize(self, data):
        return (data - 0.5) / 0.5


class NumpyDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True
    ):
        super().__init__(
            data_dir,
            target_modality,
            source_modality,
            stage,
            image_size,
            norm,
            padding
        )

        # Load target images
        self.target = self._load_data(self.target_modality)
        self.source = self._load_data(self.source_modality)

        # Get original shape
        self.original_shape = self.target.shape[-2:]

        # Load subject ids
        self.subject_ids = self._load_subject_ids('subject_ids.yaml')

        # Padding
        if self.padding:
            self.target = self._pad_data(self.target)
            self.source = self._pad_data(self.source)

        # Normalize
        if self.norm:
            self.target = self._normalize(self.target)
            self.source = self._normalize(self.source)

        # Expand channel dim
        self.target = np.expand_dims(self.target, axis=1)
        self.source = np.expand_dims(self.source, axis=1)

    def _load_data(self, contrast):
        data_dir = os.path.join(self.data_dir, contrast, self.stage)
        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

        # Sort by slice index
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        data = []
        for file in files:
            data.append(np.load(os.path.join(data_dir, file)))
        
        return np.array(data).astype(np.float32)

    def _load_subject_ids(self, filename):
        subject_ids_path = os.path.join(self.data_dir, filename)
        if os.path.exists(subject_ids_path):
            with open(subject_ids_path, 'r') as f:
                subject_ids = np.array(yaml.load(f, Loader=yaml.FullLoader))
        else:
            subject_ids = None

        return subject_ids

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        return self.target[i], self.source[i], i


class DataModule(L.LightningDataModule):
    def __init__(
        self, 
        dataset_dir,
        source_modality,
        target_modality,
        dataset_class,
        image_size,
        padding,
        norm,
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.image_size = image_size
        self.padding = padding
        self.norm = norm
        self.num_workers = num_workers

        self.dataset_class = globals()[dataset_class]

    def setup(self, stage: str) -> None:
        target_modality = self.target_modality
        source_modality = self.source_modality

        if stage == "fit":
            self.train_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='train',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm
            )

            self.val_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='val',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='test',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
