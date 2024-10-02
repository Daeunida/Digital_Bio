import os
import pandas as pd
import pydicom  # DICOM 파일을 처리하기 위한 라이브러리
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# CT 이미지 전처리 함수
def preprocess_ct_images(ct_slices, target_depth):
    num_slices = ct_slices.shape[0]  # CT 슬라이스의 개수를 확인
    if num_slices < target_depth:  # 슬라이스가 적으면 패딩을 추가하여 타겟 깊이 맞춤
        padding = (target_depth - num_slices) // 2
        ct_slices = np.pad(ct_slices, ((padding, padding), (0, 0), (0, 0)), 'constant')  # 슬라이스 중앙에 패딩 추가
        ct_slices = torch.from_numpy(ct_slices)  # NumPy 배열을 Torch 텐서로 변환
    elif num_slices > target_depth:  # 슬라이스가 많으면 중앙을 기준으로 타겟 깊이만큼 슬라이스를 자름
        start = (num_slices - target_depth) // 2
        ct_slices = ct_slices[start:start + target_depth]
    return ct_slices

# 데이터셋을 train/valid/test로 나누는 함수 (클래스 0과 1이 모두 포함되도록 stratify 적용)
def split_dataset(annotations_file, train_size=0.8, valid_size=0.1, test_size=0.1, random_state=42):
    
    annotations = pd.read_csv(annotations_file) # 주어진 파일에서 주석(annotations) 데이터를 읽음
    
    # 'Survival Status' 컬럼이 클래스 레이블을 나타낸다고 가정하고 stratify를 위해 사용
    # stratify_col = annotations['Survival Status']
    stratify_col = annotations['']
    
    
    # 데이터를 train, valid, test로 분리 (stratify로 클래스 비율 유지)
    train_data, temp_data = train_test_split(annotations, train_size=train_size, stratify=stratify_col, random_state=random_state)
    valid_data, test_data = train_test_split(temp_data, test_size=test_size/(valid_size+test_size), stratify=temp_data['Survival Status'], random_state=random_state)
    
    return train_data, valid_data, test_data

# NSCLC (비소세포 폐암) 데이터를 처리하기 위한 데이터셋 클래스
class NSCLCDataset(Dataset):
    def __init__(self, annotations, transform=None, target_depth=257):
        self.annotations = annotations  # 주석 데이터 (파일 경로, 임상 데이터, 레이블 등)
        self.transform = transform  # 이미지 변환을 위한 transform
        self.target_depth = target_depth  # 목표로 하는 CT 슬라이스 깊이
        self.clinical_features = self.annotations.columns[1:-2]  # 임상 데이터의 특징을 정의 (주석 파일의 특정 열 사용)

    def __len__(self):
        return len(self.annotations)  # 데이터셋의 샘플 수 반환

    def __getitem__(self, idx):
        dir_path = self.annotations.iloc[idx]['File Location']  # DICOM 파일이 저장된 경로를 가져옴
        img_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.dcm')]  # 해당 경로에서 DICOM 파일들을 읽음

        images = []
        for img_file in sorted(img_files):  # 파일 이름을 정렬하여 순서대로 읽음
            img = pydicom.dcmread(img_file).pixel_array  # DICOM 파일에서 이미지 데이터를 가져옴
            
            if len(img.shape) == 2:  # 2D 배열이면, 즉 그레이스케일 이미지라면
                img = np.expand_dims(img, axis=0)  # [H, W] -> [1, H, W]로 확장하여 채널 추가 (그레이스케일이므로 1채널)
            
            img = torch.tensor(img, dtype=torch.float32)  # 텐서로 변환 (ToTensor 대신 사용)
            
            if self.transform:  # transform이 정의된 경우 이미지에 변환 적용
                img = self.transform(img)
            
            images.append(img)

        images = torch.stack(images).squeeze()  # 모든 이미지를 하나의 텐서로 결합
        images = preprocess_ct_images(images, self.target_depth)  # CT 이미지를 전처리 (슬라이스 깊이 맞추기)
        images = images.unsqueeze(0).type(torch.float32)  # 이미지를 [1, H, W, D]로 reshape 후 float32 타입으로 변환

        clinical_data = self.annotations.iloc[idx][self.clinical_features].values.astype(np.float32)  # 임상 데이터를 float32로 변환
        clinical_data = torch.tensor(clinical_data)  # 임상 데이터를 텐서로 변환

        label = torch.tensor(self.annotations.iloc[idx]['Survival Status'], dtype=torch.long)  # 생존 상태(라벨)를 텐서로 변환
        return images, clinical_data, label  # 이미지, 임상 데이터, 라벨을 반환

# 데이터 로더를 반환하는 함수
def get_data_loaders(annotations_file, batch_size=2, num_workers=16):
    
    # 데이터를 train, valid, test로 나눔
    train_data, valid_data, test_data = split_dataset(annotations_file)
    
    # 이미지 변환을 위한 transform 정의
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img)  # ToTensor를 대신하여 이미지에 추가 변환 작업 없음
    ])

    # 데이터셋 생성
    train_dataset = NSCLCDataset(annotations=train_data, transform=transform)  # 학습 데이터셋 생성
    valid_dataset = NSCLCDataset(annotations=valid_data, transform=transform)  # 검증 데이터셋 생성
    test_dataset = NSCLCDataset(annotations=test_data, transform=transform)  # 테스트 데이터셋 생성

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=True, batch_size=batch_size)  # 학습 데이터 로더
    valid_loader = DataLoader(valid_dataset, num_workers=num_workers, shuffle=False, batch_size=1)  # 검증 데이터 로더
    test_loader = DataLoader(test_dataset, num_workers=num_workers, shuffle=False, batch_size=1)  # 테스트 데이터 로더

    return train_loader, valid_loader, test_loader  # 데이터 로더 반환

