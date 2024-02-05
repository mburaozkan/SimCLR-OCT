import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class OCTDataset(Dataset):
    def __init__(self, directory, patient_numbers, mm, transform=None, label=False):
        self.base_path = directory
        self.patient_numbers = patient_numbers
        self.mm = mm
        self.transform = transform
        # self.projection_map_headers = ["FULL", "ILM_OPL", "OPL_BM"]
        self.label = label

        if mm == 3:
            # width - height
            self.shape = (304, 640)
        elif mm == 6:
            # width - height
            self.shape = (400, 640)

        # Total number of elements to skip at the beginning and end
        self.skip_start = 80
        self.skip_end = 80
        self.total_scans_per_patient = 304
        self.valid_scans_per_patient = self.total_scans_per_patient - self.skip_start - self.skip_end

    def load_b_scans(self, patient_number, scan_number):
        folder_path = os.path.join(self.base_path, "OCT", str(patient_number))
        dir_ = os.listdir(folder_path)
        # Delete files that are in checked_paths
        # dir_ = [f for f in dir_ if f not in checked_paths]
        # dir_ = dir_[:1]
        if os.path.isdir(folder_path):
            # for f in sorted([int(fn.split(".")[0]) for fn in os.listdir(folder_path)]):            
            i = 0
            for f in sorted([int(fn.split(".")[0]) for fn in dir_]):
                if i < scan_number:
                    i += 1
                    continue
                f = str(f) + ".bmp"
                # checked_paths.append(f)
                image_path = os.path.join(folder_path, f)
                image = Image.open(image_path)
                # yeni eklendi
                # if image.mode != 'RGB':
                #     image = image.convert('RGB')
        return np.array(image)
    
    def load_labels(self,patient_number):

        # Path to the Excel file
        labels_path = os.path.join(self.base_path, "Text labels.xlsx")
        # labels_path = '/home/mustafa/Project-Git/OCTA-500/OCTA_3mm/Text labels.xlsx'
        # Read the Excel file
        df = pd.read_excel(labels_path)
        if 'Disease_Num' not in df.columns:
            # Create a mapping for diseases to numbers
            unique_diseases = df['Disease'].unique()
            disease_to_num = {disease: i for i, disease in enumerate(unique_diseases)}

            # Map each disease to its corresponding number
            df['Disease_Num'] = df['Disease'].map(disease_to_num)

            # Save the updated dataframe back to the Excel file
            df.to_excel(labels_path, index=False)
        # Assuming the Excel file has columns 'ID' for patient number and 'Label' for labels
        label = df[df['ID'] == patient_number]['Disease_Num'].values[0]

        return label
    
    def __len__(self):
        return len(self.patient_numbers) * self.valid_scans_per_patient

    def __getitem__(self, idx):
        patient_index = idx // self.valid_scans_per_patient
        scan_offset = idx % self.valid_scans_per_patient
        patient_number = self.patient_numbers[patient_index]
        scan_number = self.skip_start + scan_offset
        if self.label:
            label = self.load_labels(patient_number)
        b_scan = self.load_b_scans(patient_number, scan_number)

        # Apply the transformation twice to get two augmented views of the same B-scan
        b_scan_view1 = self.transform(Image.fromarray(b_scan))
        b_scan_view2 = self.transform(Image.fromarray(b_scan))
        b_scan = Image.fromarray(b_scan)

        # Ensure the output shapes are consistent
        if b_scan_view1.shape != b_scan_view2.shape:
            raise ValueError(f"Shape mismatch: {b_scan_view1.shape} vs {b_scan_view2.shape}")

        if self.label:
            if label != 0:
                label = 1
            return (b_scan_view1, b_scan_view2), (1 - label, 1 - label)
        return b_scan_view1, b_scan_view2