<<<<<<< HEAD
import csv
import torch
import numpy as np
import random
import os
import enum
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

class DataLoaderType(enum.Enum):
    POCKET = 1,
    WRIST = 2,
    POCKET_AND_WRIST = 3,
    POCKET_OR_WRIST = 4
    
class DataType(enum.Enum):
    TRAIN = 1,
    VALIDATION = 2,
    TEST = 3

class CustomDataset(Dataset):
    def __init__(self, generate_new_dataset:bool, description_file_path:str = "", data_directory_path:str = "", data_from_samples_ratio:int = -1, data_lenght:int = -1, random_state:int = 0, mode:DataLoaderType = DataLoaderType.POCKET_AND_WRIST, dataset_directory:str = "", debug:bool = True):
        self.files_to_read_list:list = ["acce", "game_rv", "gyro", "linacce"]#"gyro_bias", "magnet", "magnet_bias", "pressure", "rv"]
        self.debug = debug
        self.mode = mode
        
        self.data_from_samples_ratio = data_from_samples_ratio
        self.data_lenght = data_lenght
        self.mapping = {
            None: 0,
            'none': 0,
            'green': 1,
            'black': 2,
            'red': 3,
            'orange': 4
        }
        
        if generate_new_dataset:
            if random_state == 0: random_state = random.randint(0, 100)
            self.sample_names_array:np.ndarray = self.read_description_file(description_file_path)
            self.train_file_name, self.val_file_name, self.test_file_name = self.shufle_and_split_samples(self.sample_names_array, random_state = random_state)
            
            self.train_sensor_reads_with_labels = self.read_all_samples(self.train_file_name, data_directory_path, self.mode, "train")
            self.val_sensor_reads_with_labels = self.read_all_samples(self.val_file_name, data_directory_path, self.mode, "val")
            self.test_sensor_reads_with_labels = self.read_all_samples(self.test_file_name, data_directory_path, self.mode, "test")
            
            self.train_data, self.train_labels = self.convert_samples_to_data(self.train_sensor_reads_with_labels, self.data_lenght)
            self.val_data, self.val_labels = self.convert_samples_to_data(self.val_sensor_reads_with_labels, self.data_lenght)
            self.test_data, self.test_labels = self.convert_samples_to_data(self.test_sensor_reads_with_labels, self.data_lenght)
            print(self.train_data.shape)
            print(self.train_labels.shape)
            
            
            
            train = {
                'data': self.train_data,
                'labels': self.train_labels
            }
            val = {
                'data': self.val_data,
                'labels': self.val_labels
            }
            test = {
                'data': self.test_data,
                'labels': self.test_labels
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join("Data\Tensory", self.mode.name+timestamp)
            os.makedirs(result_path, exist_ok=True)
            torch.save(train, os.path.join(result_path, "train.pt"))
            torch.save(val, os.path.join(result_path, "val.pt"))
            torch.save(test, os.path.join(result_path, "test.pt"))
            
            mean, std = self.compute_channel_stats(self.train_data.numpy())
            
        else:
            assert dataset_directory != "", "Bledna sciezka do danych"
            
            os.path.isfile(os.path.join(dataset_directory, "train.pt"))
            os.path.isfile(os.path.join(dataset_directory, "val.pt"))
            os.path.isfile(os.path.join(dataset_directory, "test.pt"))   
            
            loaded_train = torch.load(os.path.join(dataset_directory , 'train.pt'))
            self.train_data = loaded_train['data']
            self.train_labels = loaded_train['labels']
            loaded_val = torch.load(os.path.join(dataset_directory , "val.pt"))
            self.val_data = loaded_val['data']
            self.val_labels = loaded_val['labels']
            loaded_test = torch.load(os.path.join(dataset_directory , "test.pt"))
            self.test_data = loaded_test['data']
            self.test_labels = loaded_test['labels']
            
            assert self.train_data.shape[0] == self.train_labels.shape[0], "Błedne dane treningowe"
            assert self.test_data.shape[0] == self.test_labels.shape[0], "Błedne dane testowe"
            assert self.val_data.shape[0] == self.val_labels.shape[0], "Błedne dane walidacyjne"
       
    def set_datatype(self, type:DataType):
        self.datatype = type
         
    def __len__(self):
        if self.datatype == DataType.TRAIN:
            return len(self.train_data)
        if self.datatype == DataType.VALIDATION:
            return len(self.val_data)
        if self.datatype == DataType.TEST:
            return len(self.test_data)

    def __getitem__(self, idx,):
        if self.datatype == DataType.TRAIN:
            sample = self.train_data[idx]
            label = self.train_labels[idx]
        if self.datatype == DataType.VALIDATION:
            sample = self.val_data[idx]
            label = self.val_labels[idx]
        if self.datatype == DataType.TEST:
            sample = self.test_data[idx]
            label = self.test_labels[idx]
        
        # if self.transform:
        #     sample = self.transform(sample)

        return sample, label
       
    def compute_channel_stats(self, data):
        mean = data.mean(axis=(0, 1))
        std = data.std(axis=(0, 1))
        return mean, std   
        
    def read_all_samples(self, samples_array:np.ndarray, basic_directory:str, mode:DataLoaderType = DataLoaderType.POCKET_AND_WRIST, name:str = "default name")-> tuple:
        result = []
        result_label = []
        choosed_by_mode_samples_array = np.array([])
        if mode == DataLoaderType.WRIST:
            choosed_by_mode_samples_array = samples_array[:, [1, 3, 4, 5]]
            for sample, label, start, stop in choosed_by_mode_samples_array:
                sample_path = os.path.join(basic_directory, "Telefon_na_rece", sample)
                if self.debug: print(sample_path, " ", label) 
                all_sensors_read = self.read_all_sensors(sample_path, int(start), int(stop))
                if all_sensors_read.numel() > 0:  # Make sure it's not empty
                    result.append(all_sensors_read)
                    result_label.append(label)
            print(mode)
        elif mode == DataLoaderType.POCKET:
            choosed_by_mode_samples_array = samples_array[:, [2, 3, 4, 5]]
            for sample, label, start, stop in choosed_by_mode_samples_array:
                sample_path = os.path.join(basic_directory, "Telefon_w_kieszeni", sample)
                if self.debug: print(sample_path, " ", label) 
                all_sensors_read = self.read_all_sensors(sample_path, int(start), int(stop))
                if all_sensors_read.numel() > 0:  # Make sure it's not empty
                    result.append(all_sensors_read)
                    result_label.append(label)
            print(mode)
        elif mode == DataLoaderType.POCKET_AND_WRIST or mode == DataLoaderType.POCKET_OR_WRIST:
            choosed_by_mode_samples_array = samples_array[:, 1:6]
            for sample1, sample2, label, start, stop in choosed_by_mode_samples_array:
                sample_path1 = os.path.join(basic_directory, "Telefon_na_rece", sample1)
                sample_path2 = os.path.join(basic_directory, "Telefon_w_kieszeni", sample2)
                if self.debug: print(sample_path1, " ", sample_path2, " ", label) 
                all_sensors_read1 = self.read_all_sensors(sample_path1, int(start), int(stop))
                all_sensors_read2 = self.read_all_sensors(sample_path2, int(start), int(stop))
                
                if all_sensors_read1.numel() > 0 and all_sensors_read2.numel() > 0:  # Make sure it's not empty
                    if mode == DataLoaderType.POCKET_OR_WRIST:
                        result.append(all_sensors_read1)
                        result.append(all_sensors_read2)
                        result_label.append(label)
                        result_label.append(label)
                    if mode == DataLoaderType.POCKET_AND_WRIST:
                        raise Exception("This mode is not implemented yet. Due to different start moments for wrist and pocket")
            print(mode)
            
        if result:
            print(f"[{name}] Samples number: {len(result)}")
        else:
            result = torch.empty(0)
            print("[{name}] No valid data found across sessions.")
        
        return list(zip(result, result_label))
               
    def read_all_sensors(self, basic_path:str, bias_head:int, bias_tail:int):
        all_sensor_data = []

        for sensor in self.files_to_read_list:
            file_path = os.path.join(basic_path, sensor + ".txt")
            if self.debug: print(f"Reading file: {file_path}")

            try:
                raw_lines = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue  # Skip metadata/header lines
                        raw_lines.append(line.strip())

                # Apply head/tail bias
                raw_lines = raw_lines[bias_head:]
                if bias_tail > 0:
                    raw_lines = raw_lines[:-bias_tail]

                sensor_data = []
                for line in raw_lines:
                    parts = line.split()
                    if len(parts) < 2:
                        if self.debug:print(f"Skipping malformed line in {file_path}: {line}")
                        continue
                    try:
                        values = list(map(float, parts[1:]))  # Skip timestamp
                        sensor_data.append(values)
                    except ValueError as ve:
                        print(f"Skipping non-numeric line in {file_path}: {line} - {ve}")
                        continue

                all_sensor_data.append(sensor_data)

            except FileNotFoundError:
                print(f"[ERROR] File not found: {file_path}")
                return torch.empty(0)
            except Exception as e:
                print(f"[ERROR] Failed to read {file_path}: {e}")
                return torch.empty(0)

        # Combine sensor rows across files
        try:
            combined_rows = [
                sum(row_sets, [])  # Flatten values from all sensors for the same row
                for row_sets in zip(*all_sensor_data)
            ]
        except Exception as e:
            print(f"[ERROR] Failed to combine sensor data: {e}")
            return torch.empty(0)

        tensor = torch.tensor(combined_rows)
        if self.debug: print(f"Final tensor shape: {tensor.shape}")
        return tensor
            
    def convert_samples_to_data(self, list_of_samples_with_labels: list, length: int):
        data_list = []
        label_list = []
        print("Ratio:", self.data_from_samples_ratio)
        
        for i in range(self.data_from_samples_ratio):
            print(f"Round {i+1}/{self.data_from_samples_ratio}")
            for sample, label in list_of_samples_with_labels:
                if sample.shape[0] < length:
                    print("Skipping sample - too short:", sample.shape)
                    continue

                random_moment = random.randint(0, sample.shape[0] - length)
                chunk = sample[random_moment:random_moment + length]

                data_list.append(chunk)
                if self.debug: print(len(data_list))
                label_list.append(self.color_str_to_int(label))

            print(f"Accumulated samples: {len(data_list)}")

        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list)

        return data_tensor, label_tensor

    def color_str_to_int(self, color):
        if isinstance(color, np.str_):
            color = str(color).lower()
        elif isinstance(color, str):
            color = color.lower()

        return self.mapping.get(color, 0) 
    
    def read_description_file(self, file_path:str)-> np.ndarray:
        try:
            with open(file_path, newline='', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=';')  # Change delimiter if needed
                headers = next(reader) 
                reader_list = [row for row in reader] 
                reader_array = np.array(reader_list)
                return reader_array
        except Exception as e:
            print(f"Błąd podczas odczytu pliku: {e}")

    def shufle_and_split_samples(self, data: np.ndarray, train_size=0.7, val_size=0.15, random_state=0):
        """
        Shuffles and splits the given NumPy array into training, validation, and test sets.
        
        :param data: NumPy array containing data
        :param train_size: Proportion of data for training
        :param val_size: Proportion of data for validation
        :param test_size: Proportion of data for testing
        :param random_state: Random seed for reproducibility
        :return: train, val, test NumPy arrays
        """
        assert train_size - val_size < 1, "Train and validation size must be smaller or equal to 1"
        
        test_size = 1 - train_size - val_size 
        if random_state == 0:
            random_state = random.randint(0, 100)

        np.random.seed(random_state)
        np.random.shuffle(data)

        train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=random_state)
        val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=random_state)

        return train_data, val_data, test_data
    

if __name__ == "__main__":
    dataset = CustomDataset(True, "Data\\opis_przejsc.csv", "C:\\Users\\Marcin\\Desktop\\Studia\\IoT\\Data", data_from_samples_ratio=50, data_lenght = 400 , random_state = 42, mode = DataLoaderType.POCKET, dataset_directory =r"C:\Users\Marcin\Desktop\Studia\IoT\Data\Tensory\POCKET20250427_172800", debug=False)
    print("-------------------------------------")

=======
import csv
import torch
import numpy as np
import random
import os
import enum
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

class DataLoaderType(enum.Enum):
    POCKET = 1,
    WRIST = 2,
    POCKET_AND_WRIST = 3,
    POCKET_OR_WRIST = 4
    
class DataType(enum.Enum):
    TRAIN = 1,
    VALIDATION = 2,
    TEST = 3

class CustomDataset(Dataset):
    def __init__(self, generate_new_dataset:bool, description_file_path:str = "", data_directory_path:str = "", data_from_samples_ratio:int = -1, data_lenght:int = -1, random_state:int = 0, mode:DataLoaderType = DataLoaderType.POCKET_AND_WRIST, dataset_directory:str = "", debug:bool = True):
        self.files_to_read_list:list = ["acce", "game_rv", "gyro", "linacce"]#"gyro_bias", "magnet", "magnet_bias", "pressure", "rv"]
        self.debug = debug
        self.mode = mode
        
        self.data_from_samples_ratio = data_from_samples_ratio
        self.data_lenght = data_lenght
        self.mapping = {
            None: 0,
            'none': 0,
            'green': 1,
            'black': 2,
            'red': 3,
            'orange': 4
        }
        
        if generate_new_dataset:
            if random_state == 0: random_state = random.randint(0, 100)
            self.sample_names_array:np.ndarray = self.read_description_file(description_file_path)
            self.train_file_name, self.val_file_name, self.test_file_name = self.shufle_and_split_samples(self.sample_names_array, random_state = random_state)
            
            self.train_sensor_reads_with_labels = self.read_all_samples(self.train_file_name, data_directory_path, self.mode, "train")
            self.val_sensor_reads_with_labels = self.read_all_samples(self.val_file_name, data_directory_path, self.mode, "val")
            self.test_sensor_reads_with_labels = self.read_all_samples(self.test_file_name, data_directory_path, self.mode, "test")
            
            self.train_data, self.train_labels = self.convert_samples_to_data(self.train_sensor_reads_with_labels, self.data_lenght)
            self.val_data, self.val_labels = self.convert_samples_to_data(self.val_sensor_reads_with_labels, self.data_lenght)
            self.test_data, self.test_labels = self.convert_samples_to_data(self.test_sensor_reads_with_labels, self.data_lenght)
            print(self.train_data.shape)
            print(self.train_labels.shape)
            
            
            
            train = {
                'data': self.train_data,
                'labels': self.train_labels
            }
            val = {
                'data': self.val_data,
                'labels': self.val_labels
            }
            test = {
                'data': self.test_data,
                'labels': self.test_labels
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join("Data\Tensory", self.mode.name+timestamp)
            os.makedirs(result_path, exist_ok=True)
            torch.save(train, os.path.join(result_path, "train.pt"))
            torch.save(val, os.path.join(result_path, "val.pt"))
            torch.save(test, os.path.join(result_path, "test.pt"))
            
            mean, std = self.compute_channel_stats(self.train_data.numpy())
            
        else:
            assert dataset_directory != "", "Bledna sciezka do danych"
            
            os.path.isfile(os.path.join(dataset_directory, "train.pt"))
            os.path.isfile(os.path.join(dataset_directory, "val.pt"))
            os.path.isfile(os.path.join(dataset_directory, "test.pt"))   
            
            loaded_train = torch.load(os.path.join(dataset_directory , 'train.pt'))
            self.train_data = loaded_train['data']
            self.train_labels = loaded_train['labels']
            loaded_val = torch.load(os.path.join(dataset_directory , "val.pt"))
            self.val_data = loaded_val['data']
            self.val_labels = loaded_val['labels']
            loaded_test = torch.load(os.path.join(dataset_directory , "test.pt"))
            self.test_data = loaded_test['data']
            self.test_labels = loaded_test['labels']
            
            assert self.train_data.shape[0] == self.train_labels.shape[0], "Błedne dane treningowe"
            assert self.test_data.shape[0] == self.test_labels.shape[0], "Błedne dane testowe"
            assert self.val_data.shape[0] == self.val_labels.shape[0], "Błedne dane walidacyjne"
       
    def set_datatype(self, type:DataType):
        self.datatype = type
         
    def __len__(self):
        if self.datatype == DataType.TRAIN:
            return len(self.train_data)
        if self.datatype == DataType.VALIDATION:
            return len(self.val_data)
        if self.datatype == DataType.TEST:
            return len(self.test_data)

    def __getitem__(self, idx,):
        if self.datatype == DataType.TRAIN:
            sample = self.train_data[idx]
            label = self.train_labels[idx]
        if self.datatype == DataType.VALIDATION:
            sample = self.val_data[idx]
            label = self.val_labels[idx]
        if self.datatype == DataType.TEST:
            sample = self.test_data[idx]
            label = self.test_labels[idx]
        
        # if self.transform:
        #     sample = self.transform(sample)

        return sample, label
       
    def compute_channel_stats(self, data):
        mean = data.mean(axis=(0, 1))
        std = data.std(axis=(0, 1))
        return mean, std   
        
    def read_all_samples(self, samples_array:np.ndarray, basic_directory:str, mode:DataLoaderType = DataLoaderType.POCKET_AND_WRIST, name:str = "default name")-> tuple:
        result = []
        result_label = []
        choosed_by_mode_samples_array = np.array([])
        if mode == DataLoaderType.WRIST:
            choosed_by_mode_samples_array = samples_array[:, [1, 3, 4, 5]]
            for sample, label, start, stop in choosed_by_mode_samples_array:
                sample_path = os.path.join(basic_directory, "Telefon_na_rece", sample)
                if self.debug: print(sample_path, " ", label) 
                all_sensors_read = self.read_all_sensors(sample_path, int(start), int(stop))
                if all_sensors_read.numel() > 0:  # Make sure it's not empty
                    result.append(all_sensors_read)
                    result_label.append(label)
            print(mode)
        elif mode == DataLoaderType.POCKET:
            choosed_by_mode_samples_array = samples_array[:, [2, 3, 4, 5]]
            for sample, label, start, stop in choosed_by_mode_samples_array:
                sample_path = os.path.join(basic_directory, "Telefon_w_kieszeni", sample)
                if self.debug: print(sample_path, " ", label) 
                all_sensors_read = self.read_all_sensors(sample_path, int(start), int(stop))
                if all_sensors_read.numel() > 0:  # Make sure it's not empty
                    result.append(all_sensors_read)
                    result_label.append(label)
            print(mode)
        elif mode == DataLoaderType.POCKET_AND_WRIST or mode == DataLoaderType.POCKET_OR_WRIST:
            choosed_by_mode_samples_array = samples_array[:, 1:6]
            for sample1, sample2, label, start, stop in choosed_by_mode_samples_array:
                sample_path1 = os.path.join(basic_directory, "Telefon_na_rece", sample1)
                sample_path2 = os.path.join(basic_directory, "Telefon_w_kieszeni", sample2)
                if self.debug: print(sample_path1, " ", sample_path2, " ", label) 
                all_sensors_read1 = self.read_all_sensors(sample_path1, int(start), int(stop))
                all_sensors_read2 = self.read_all_sensors(sample_path2, int(start), int(stop))
                
                if all_sensors_read1.numel() > 0 and all_sensors_read2.numel() > 0:  # Make sure it's not empty
                    if mode == DataLoaderType.POCKET_OR_WRIST:
                        result.append(all_sensors_read1)
                        result.append(all_sensors_read2)
                        result_label.append(label)
                        result_label.append(label)
                    if mode == DataLoaderType.POCKET_AND_WRIST:
                        raise Exception("This mode is not implemented yet. Due to different start moments for wrist and pocket")
            print(mode)
            
        if result:
            print(f"[{name}] Samples number: {len(result)}")
        else:
            result = torch.empty(0)
            print("[{name}] No valid data found across sessions.")
        
        return list(zip(result, result_label))
               
    def read_all_sensors(self, basic_path:str, bias_head:int, bias_tail:int):
        all_sensor_data = []

        for sensor in self.files_to_read_list:
            file_path = os.path.join(basic_path, sensor + ".txt")
            if self.debug: print(f"Reading file: {file_path}")

            try:
                raw_lines = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue  # Skip metadata/header lines
                        raw_lines.append(line.strip())

                # Apply head/tail bias
                raw_lines = raw_lines[bias_head:]
                if bias_tail > 0:
                    raw_lines = raw_lines[:-bias_tail]

                sensor_data = []
                for line in raw_lines:
                    parts = line.split()
                    if len(parts) < 2:
                        if self.debug:print(f"Skipping malformed line in {file_path}: {line}")
                        continue
                    try:
                        values = list(map(float, parts[1:]))  # Skip timestamp
                        sensor_data.append(values)
                    except ValueError as ve:
                        print(f"Skipping non-numeric line in {file_path}: {line} - {ve}")
                        continue

                all_sensor_data.append(sensor_data)

            except FileNotFoundError:
                print(f"[ERROR] File not found: {file_path}")
                return torch.empty(0)
            except Exception as e:
                print(f"[ERROR] Failed to read {file_path}: {e}")
                return torch.empty(0)

        # Combine sensor rows across files
        try:
            combined_rows = [
                sum(row_sets, [])  # Flatten values from all sensors for the same row
                for row_sets in zip(*all_sensor_data)
            ]
        except Exception as e:
            print(f"[ERROR] Failed to combine sensor data: {e}")
            return torch.empty(0)

        tensor = torch.tensor(combined_rows)
        if self.debug: print(f"Final tensor shape: {tensor.shape}")
        return tensor
            
    def convert_samples_to_data(self, list_of_samples_with_labels: list, length: int):
        data_list = []
        label_list = []
        print("Ratio:", self.data_from_samples_ratio)
        
        for i in range(self.data_from_samples_ratio):
            print(f"Round {i+1}/{self.data_from_samples_ratio}")
            for sample, label in list_of_samples_with_labels:
                if sample.shape[0] < length:
                    print("Skipping sample - too short:", sample.shape)
                    continue

                random_moment = random.randint(0, sample.shape[0] - length)
                chunk = sample[random_moment:random_moment + length]

                data_list.append(chunk)
                if self.debug: print(len(data_list))
                label_list.append(self.color_str_to_int(label))

            print(f"Accumulated samples: {len(data_list)}")

        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list)

        return data_tensor, label_tensor

    def color_str_to_int(self, color):
        if isinstance(color, np.str_):
            color = str(color).lower()
        elif isinstance(color, str):
            color = color.lower()

        return self.mapping.get(color, 0) 
    
    def read_description_file(self, file_path:str)-> np.ndarray:
        try:
            with open(file_path, newline='', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter=';')  # Change delimiter if needed
                headers = next(reader) 
                reader_list = [row for row in reader] 
                reader_array = np.array(reader_list)
                return reader_array
        except Exception as e:
            print(f"Błąd podczas odczytu pliku: {e}")

    def shufle_and_split_samples(self, data: np.ndarray, train_size=0.7, val_size=0.15, random_state=0):
        """
        Shuffles and splits the given NumPy array into training, validation, and test sets.
        
        :param data: NumPy array containing data
        :param train_size: Proportion of data for training
        :param val_size: Proportion of data for validation
        :param test_size: Proportion of data for testing
        :param random_state: Random seed for reproducibility
        :return: train, val, test NumPy arrays
        """
        assert train_size - val_size < 1, "Train and validation size must be smaller or equal to 1"
        
        test_size = 1 - train_size - val_size 
        if random_state == 0:
            random_state = random.randint(0, 100)

        np.random.seed(random_state)
        np.random.shuffle(data)

        train_data, temp_data = train_test_split(data, test_size=(val_size + test_size), random_state=random_state)
        val_data, test_data = train_test_split(temp_data, test_size=(test_size / (val_size + test_size)), random_state=random_state)

        return train_data, val_data, test_data
    

if __name__ == "__main__":
    dataset = CustomDataset(True, "Data\\opis_przejsc.csv", "C:\\Users\\Marcin\\Desktop\\Studia\\IoT\\Data", data_from_samples_ratio=50, data_lenght = 400 , random_state = 42, mode = DataLoaderType.POCKET, dataset_directory =r"C:\Users\Marcin\Desktop\Studia\IoT\Data\Tensory\POCKET20250427_172800", debug=False)
    print("-------------------------------------")

>>>>>>> 7269691ad6e9e8e4ef1f57dda3e82142c568ee09
    