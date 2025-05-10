import os
import torch
from torch.utils.data import Dataset
import torchaudio

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.label_to_idx = {}
        self.speaker_to_idx = {}
        label_idx_counter = 0
        speaker_idx_counter = 0

        for subdir in os.listdir(root_dir):
            if not subdir.endswith("_splitted"):
                continue
            speaker = subdir.replace("_splitted", "")
            speaker_dir = os.path.join(root_dir, subdir)

            if speaker not in self.speaker_to_idx:
                self.speaker_to_idx[speaker] = speaker_idx_counter
                speaker_idx_counter += 1

            for file in os.listdir(speaker_dir):
                if not file.endswith(".wav"):
                    continue
                parts = file.replace(".wav", "").split("_")
                if len(parts) < 3:
                    continue  # ignore unexpected filenames
                _, label, _ = parts

                if label not in self.label_to_idx:
                    self.label_to_idx[label] = label_idx_counter
                    label_idx_counter += 1

                self.samples.append((
                    os.path.join(speaker_dir, file),
                    speaker,
                    label
                ))

        self.num_speakers = len(self.speaker_to_idx)
        self.num_labels = len(self.label_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, speaker, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(path)
        if self.transform:
            waveform = self.transform(waveform)

        speaker_index = self.speaker_to_idx[speaker]
        label_index = self.label_to_idx[label]
        
        # speaker_onehot = torch.nn.functional.one_hot(torch.tensor(speaker_index), num_classes=self.num_speakers).float()
        # label_onehot = torch.nn.functional.one_hot(torch.tensor(label_index), num_classes=self.num_labels).float()

        return waveform, speaker_index, label_index
    
    def save_splits(self, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        waveforms_to_padded = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            waveforms_to_padded.append(waveform)
        waveforms_padded = self.pad_to_longest(waveforms_to_padded)
            
        data = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            data.append((waveforms_padded[i], speaker_vec, label_vec))


        torch.manual_seed(42)  # deterministyczny shuffle
        data = [data[i] for i in torch.randperm(len(data))]

        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set = data[:n_train]
        val_set = data[n_train:n_train + n_val]
        test_set = data[n_train + n_val:]

        os.makedirs(output_dir, exist_ok=True)

        torch.save(train_set, os.path.join(output_dir, "train.pt"))
        torch.save(val_set, os.path.join(output_dir, "val.pt"))
        torch.save(test_set, os.path.join(output_dir, "test.pt"))

        print(f"Saved splits to '{output_dir}':")
        print(f"- train: {len(train_set)} samples")
        print(f"- val:   {len(val_set)} samples")
        print(f"- test:  {len(test_set)} samples")
    
    def pad_to_longest(self,waveforms: list[torch.Tensor]) -> torch.Tensor:
        max_len = max([w.shape[1] for w in waveforms])
        batch_size = len(waveforms)
        channels = waveforms[0].shape[0]

        padded = torch.zeros(batch_size, channels, max_len)

        for i, w in enumerate(waveforms):
            padded[i, :, :w.shape[1]] = w
            
        return padded

        
    def get_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        waveforms_to_padded = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            waveforms_to_padded.append(waveform)
        waveforms_padded = self.pad_to_longest(waveforms_to_padded)
            
        data = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            data.append((waveforms_padded[i], speaker_vec, label_vec))

        torch.manual_seed(42)  # deterministyczny shuffle
        data = [data[i] for i in torch.randperm(len(data))]

        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set = data[:n_train]
        val_set = data[n_train:n_train + n_val]
        test_set = data[n_train + n_val:]
        
        return train_set, val_set, test_set

if __name__ == "__main__":
    dataset_path = "Data\\VoiceCommands\\Gloskomendy"
    output_dir = "Data\\VoiceCommands\\Tensory\\1"

    dataset = AudioDataset(root_dir=dataset_path)

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Detected speakers: {len(dataset.speaker_to_idx)}")
    print(f"Detected emotion labels: {len(dataset.label_to_idx)}")

    print("\nUnique Speakers:")
    for speaker, idx in dataset.speaker_to_idx.items():
        print(f"- {speaker}: index {idx}")

    print("\nUnique Emotion Labels:")
    for label, idx in dataset.label_to_idx.items():
        print(f"- {label}: index {idx}")

    waveform, speaker_vec, emotion_vec = dataset[0]
    print(f"\nExample sample:")
    print(f"- Waveform shape: {waveform.shape}")
    print(f"- Speaker one-hot vector: {speaker_vec}")
    print(f"- Emotion one-hot vector: {emotion_vec}")

    # Save splits
    dataset.save_splits(output_dir)
import os
import torch
from torch.utils.data import Dataset
import torchaudio

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.label_to_idx = {}
        self.speaker_to_idx = {}
        label_idx_counter = 0
        speaker_idx_counter = 0

        for subdir in os.listdir(root_dir):
            if not subdir.endswith("_splitted"):
                continue
            speaker = subdir.replace("_splitted", "")
            speaker_dir = os.path.join(root_dir, subdir)

            if speaker not in self.speaker_to_idx:
                self.speaker_to_idx[speaker] = speaker_idx_counter
                speaker_idx_counter += 1

            for file in os.listdir(speaker_dir):
                if not file.endswith(".wav"):
                    continue
                parts = file.replace(".wav", "").split("_")
                if len(parts) < 3:
                    continue  # ignore unexpected filenames
                _, label, _ = parts

                if label not in self.label_to_idx:
                    self.label_to_idx[label] = label_idx_counter
                    label_idx_counter += 1

                self.samples.append((
                    os.path.join(speaker_dir, file),
                    speaker,
                    label
                ))

        self.num_speakers = len(self.speaker_to_idx)
        self.num_labels = len(self.label_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, speaker, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(path)
        if self.transform:
            waveform = self.transform(waveform)

        speaker_index = self.speaker_to_idx[speaker]
        label_index = self.label_to_idx[label]
        
        # speaker_onehot = torch.nn.functional.one_hot(torch.tensor(speaker_index), num_classes=self.num_speakers).float()
        # label_onehot = torch.nn.functional.one_hot(torch.tensor(label_index), num_classes=self.num_labels).float()

        return waveform, speaker_index, label_index
    
    def save_splits(self, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        waveforms_to_padded = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            waveforms_to_padded.append(waveform)
        waveforms_padded = self.pad_to_longest(waveforms_to_padded)
            
        data = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            data.append((waveforms_padded[i], speaker_vec, label_vec))


        torch.manual_seed(42)  # deterministyczny shuffle
        data = [data[i] for i in torch.randperm(len(data))]

        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set = data[:n_train]
        val_set = data[n_train:n_train + n_val]
        test_set = data[n_train + n_val:]

        os.makedirs(output_dir, exist_ok=True)

        torch.save(train_set, os.path.join(output_dir, "train.pt"))
        torch.save(val_set, os.path.join(output_dir, "val.pt"))
        torch.save(test_set, os.path.join(output_dir, "test.pt"))

        print(f"Saved splits to '{output_dir}':")
        print(f"- train: {len(train_set)} samples")
        print(f"- val:   {len(val_set)} samples")
        print(f"- test:  {len(test_set)} samples")
    
    def pad_to_longest(self,waveforms: list[torch.Tensor]) -> torch.Tensor:
        max_len = max([w.shape[1] for w in waveforms])
        batch_size = len(waveforms)
        channels = waveforms[0].shape[0]

        padded = torch.zeros(batch_size, channels, max_len)

        for i, w in enumerate(waveforms):
            padded[i, :, :w.shape[1]] = w
            
        return padded

        
    def get_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        waveforms_to_padded = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            waveforms_to_padded.append(waveform)
        waveforms_padded = self.pad_to_longest(waveforms_to_padded)
            
        data = []
        for i in range(len(self)):
            waveform, speaker_vec, label_vec = self[i]
            data.append((waveforms_padded[i], speaker_vec, label_vec))

        torch.manual_seed(42)  # deterministyczny shuffle
        data = [data[i] for i in torch.randperm(len(data))]

        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_set = data[:n_train]
        val_set = data[n_train:n_train + n_val]
        test_set = data[n_train + n_val:]
        
        return train_set, val_set, test_set

if __name__ == "__main__":
    dataset_path = "Data\\VoiceCommands\\Gloskomendy"
    output_dir = "Data\\VoiceCommands\\Tensory\\1"

    dataset = AudioDataset(root_dir=dataset_path)

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Detected speakers: {len(dataset.speaker_to_idx)}")
    print(f"Detected emotion labels: {len(dataset.label_to_idx)}")

    print("\nUnique Speakers:")
    for speaker, idx in dataset.speaker_to_idx.items():
        print(f"- {speaker}: index {idx}")

    print("\nUnique Emotion Labels:")
    for label, idx in dataset.label_to_idx.items():
        print(f"- {label}: index {idx}")

    waveform, speaker_vec, emotion_vec = dataset[0]
    print(f"\nExample sample:")
    print(f"- Waveform shape: {waveform.shape}")
    print(f"- Speaker one-hot vector: {speaker_vec}")
    print(f"- Emotion one-hot vector: {emotion_vec}")

    # Save splits
    dataset.save_splits(output_dir)