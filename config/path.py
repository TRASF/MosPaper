from os.path import join
from os import listdir
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent.parent
DEDICATED_UNLABELED_DIR = 'None'

cleaned_data_mix_main_dir = str(DATASET_PATH / 'dataset' / 'audios' / 'new_mix_inout')

noiseDatasetDir = str(DATASET_PATH / 'dataset' / 'audios' / 'EnvironmentalNoises' / 'cat_noise')

try:
    cleaned_data_mix_dir_files = [join(cleaned_data_mix_main_dir, file) for file in listdir(cleaned_data_mix_main_dir) if file.endswith('.wav')]
except (FileNotFoundError, NotADirectoryError):
    cleaned_data_mix_dir_files = []
try:
    noise_file_paths = [join(noiseDatasetDir, file) for file in listdir(noiseDatasetDir) if file.endswith('.wav')]
except (FileNotFoundError, NotADirectoryError):
    noise_file_paths = []

dataset = cleaned_data_mix_dir_files + noise_file_paths
