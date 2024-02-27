from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from PIL import ImageOps
import pandas as pd
import json

class VitDataset(Dataset):
    def __init__(self, root_dirs, transform=None, train=True, max_images_per_class=None):
        self.file_paths = []
        self.labels = []
        self.emotion_to_int = {
            'anger': 0, 'anxiety': 1, 'embarrass': 2,
            'happy': 3, 'pain': 4, 'sad': 5, 'normal': 6
        }

        # Initialize a dictionary to keep track of image counts per class
        self.image_counts = {emotion: 0 for emotion in self.emotion_to_int.keys()}
        self.max_images_per_class = max_images_per_class if max_images_per_class else float('inf')

        # Load data
        if train:
            dir_suffix = 'train'
        else:
            dir_suffix = 'test'

        for emotion, root_dir in root_dirs.items():
            if train:
                emotion_dir = os.path.join(root_dir, 'raw', dir_suffix)
            else:
                emotion_dir = os.path.join(root_dir)
            for file_name in os.listdir(emotion_dir):
                if self.image_counts[emotion] < self.max_images_per_class:
                    self.file_paths.append(os.path.join(emotion_dir, file_name))
                    self.labels.append(self.emotion_to_int[emotion])
                    self.image_counts[emotion] += 1
                else:
                    break 

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.check_label_distribution()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert("RGB")
        try:
            exif = image._getexif()
            orientation_key = 274  # cf. EXIF tags
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                image = ImageOps.exif_transpose(image)
        except AttributeError:
        # Image didn't have EXIF data or something else went wrong
            pass
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def check_label_distribution(self):
        print("Label distribution in dataset:")
        for label, count in self.image_counts.items():
            print(f"{label}: {count}")
            



class ResNetDataset(Dataset):
    def __init__(self, root_dirs, transform=None, train=True, max_images_per_class=None):
        self.file_paths = []
        self.labels = []
        self.emotion_to_int = {
            'anger': 0, 'anxiety': 1, 'embarrass': 2,
            'happy': 3, 'pain': 4, 'sad': 5, 'normal': 6
        }

        self.image_counts = {emotion: 0 for emotion in self.emotion_to_int.keys()}
        self.max_images_per_class = max_images_per_class if max_images_per_class else float('inf')

        # Load data
        if train:
            dir_suffix = 'train'
        else:
            dir_suffix = 'test'

        for emotion, root_dir in root_dirs.items():
            if train:
                emotion_dir = os.path.join(root_dir, 'raw', dir_suffix)
            else:
                emotion_dir = os.path.join(root_dir)
            for file_name in os.listdir(emotion_dir):
                if self.image_counts[emotion] < self.max_images_per_class:
                    self.file_paths.append(os.path.join(emotion_dir, file_name))
                    self.labels.append(self.emotion_to_int[emotion])
                    self.image_counts[emotion] += 1
                else:
                    break 

        self.transform = transform or transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.check_label_distribution()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        try:
            exif = image._getexif()
            orientation_key = 274  
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                image = ImageOps.exif_transpose(image)
        except AttributeError:
        # Image didn't have EXIF data or something else went wrong
            pass
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def check_label_distribution(self):
        print("Label distribution in dataset:")
        for label, count in self.image_counts.items():
            print(f"{label}: {count}")
            
            
            
class VitDetectionDataset(Dataset):
    def __init__(self, data_info, dict, transform=None, max_images_per_class=3000):
        self.data = []
        self.dict = dict  # Assuming dict maps labels to integers
        self.transform = transform
        self.original_sizes = []
        self.max_images_per_class = max_images_per_class
        
        # Initialize a dictionary to keep track of image counts per class
        self.image_counts = {label: 0 for label in dict.keys()}

        for df, root_dir in data_info:
            # Temporary storage to keep track of images added per class during this iteration
            temp_image_counts = {label: 0 for label in dict.keys()}
            for _, row in df.iterrows():
                label = row['faceExp_uploader']
                if temp_image_counts[label] < self.max_images_per_class:
                    file_name = row['filename']
                    image_path = os.path.join(root_dir, file_name)
                    if os.path.exists(image_path):
                        self.data.append((row, root_dir))
                        temp_image_counts[label] += 1
                        self.image_counts[label] += 1
                else:
                    continue  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row, root_dir = self.data[idx]
        file_name = row['filename']
        label = row['faceExp_uploader']
        image_path = os.path.join(root_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        
        try:
            exif = image._getexif()
            orientation_key = 274  # cf. EXIF tags
            if exif and orientation_key in exif:
                orientation = exif[orientation_key]
                image = ImageOps.exif_transpose(image)
        except AttributeError:
        # Image didn't have EXIF data or something else went wrong
            pass
        
        original_size = image.size
        self.original_sizes.append(original_size)

        if 'annot_A' in row and 'boxes' in row['annot_A']:
            coordinates = row['annot_A']['boxes']
            maxX, maxY, minX, minY = coordinates['maxX'], coordinates['maxY'], coordinates['minX'], coordinates['minY']

            if self.transform:
                image = self.transform(image)

            # Assuming transformation rescales image to a fixed size, e.g., (224, 224)
            new_size = (224, 224)
            scale_x, scale_y = new_size[0] / original_size[0], new_size[1] / original_size[1]
            dots = [maxX * scale_x, maxY * scale_y, minX * scale_x, minY * scale_y]
        else:
            # Handle case without annotation
            dots = None  # or some default value

        labels = torch.tensor(self.dict[label])
        return image, labels, dots if dots else []

    def get_original_size(self, idx):
        return self.original_sizes[idx]

    def check_label_distribution(self):
        print("Label distribution in dataset:")
        for label, count in self.image_counts.items():
            print(f"{label}: {count}")
            




def make_dataset(vit=True):
    file_paths = [
    '../data/anger/labeled/train/img_emotion_training_data(분노).json',
    '../data/anxiety/labeled/train/img_emotion_training_data(불안).json',
    '../data/embarrass/labeled/train/img_emotion_training_data(당황).json',
    '../data/happy/labeled/train/img_emotion_training_data(기쁨).json',
    '../data/pain/labeled/train/img_emotion_training_data(상처).json',
    '../data/sad/labeled/train/img_emotion_training_data(슬픔).json',
    '../data/normal/labeled/train/img_emotion_training_data(중립).json'
    ]

    # Initialize an empty DataFrame to store concatenated data
    combined_df1 = pd.DataFrame()

    # Loop through each file path, load its contents, and concatenate the data
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            combined_df1 = pd.concat([combined_df1, df],axis=0, ignore_index=True)
            
    combined_df1['faceExp_uploader'].unique()
    
    file_paths2 = [
    '../data/test_set/img_emotion_test_data(분노).json',
    '../data/test_set/img_emotion_test_data(불안).json',
    '../data/test_set/img_emotion_test_data(당황).json',
    '../data/test_set/img_emotion_test_data(기쁨).json',
    '../data/test_set/img_emotion_test_data(상처).json',
    '../data/test_set/img_emotion_test_data(슬픔).json',
    '../data/test_set/img_emotion_test_data(중립).json'
    # Add more file paths as needed
    ]

    # Initialize an empty DataFrame to store concatenated data
    combined_df2 = pd.DataFrame()

    # Loop through each file path, load its contents, and concatenate the data
    for file_path in file_paths2:
        with open(file_path, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            combined_df2 = pd.concat([combined_df2, df], axis=0, ignore_index=True)
    
    dict = {
    "분노":0,
    "불안":1,
    "당황":2,
    "기쁨":3,
    "상처":4,
    "슬픔":5,
    "중립":6
    }
    
    if (vit):
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else :
        transform =  transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    data_info = [
    (combined_df1, '../data/anger/raw/train'),
    (combined_df1, '../data/anxiety/raw/train'),
    (combined_df1, '../data/embarrass/raw/train'),
    (combined_df1, '../data/happy/raw/train'),
    (combined_df1, '../data/pain/raw/train'),
    (combined_df1, '../data/sad/raw/train'), 
    (combined_df1, '../data/normal/raw/train')   
    ]
    
    data_info2 = [
    (combined_df2, '../data/test_set1000/ang'),
    (combined_df2, '../data/test_set1000/anx'),
    (combined_df2, '../data/test_set1000/emb'),
    (combined_df2, '../data/test_set1000/hap'),
    (combined_df2, '../data/test_set1000/pai'),
    (combined_df2, '../data/test_set1000/sad'),    
    (combined_df2, '../data/test_set1000/nor'),    
    ]

    return data_info, data_info2 , transform, dict