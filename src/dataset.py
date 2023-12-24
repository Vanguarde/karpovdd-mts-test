from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_intent_dataset():
    dataset = load_dataset("AmazonScience/massive", "ru-RU").select_columns(['utt', 'intent'])
    dataset = dataset.rename_column("utt", "text")
    dataset = dataset.rename_column("intent", "label")

    return dataset