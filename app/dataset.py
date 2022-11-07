from typing import Optional

import pytorch_lightning as pl
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import BatchEncoding, DataCollatorForLanguageModeling, ViltProcessor


class ViltDataset(Dataset):
    def __init__(self, dataset: HFDataset, processor: ViltProcessor):
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor

        size = self.feature_extractor.size
        mean = self.feature_extractor.image_mean
        std = self.feature_extractor.image_std

        self.transform = T.Compose(
            [T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BatchEncoding:
        item = self.dataset[idx]
        pixel_values = self.transform(item["image"])
        inputs = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            verbose=False,
        )
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        inputs["pixel_values"] = pixel_values
        return inputs


class ViltDataModule(pl.LightningDataModule):
    def __init__(
        self, processor: ViltProcessor, batch_size: int = 32, num_workers: int = 12
    ):
        super().__init__()
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collator = DataCollatorForLanguageModeling(processor.tokenizer)

    def prepare_data(self) -> None:
        load_dataset("Bingsu/laion2b_multi_korean_subset_with_image", split="train")

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_dataset(
            "Bingsu/laion2b_multi_korean_subset_with_image", split="train"
        )
        self.train_dataset = ViltDataset(dataset, self.processor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            shuffle=True,
            num_workers=self.num_workers,
        )
