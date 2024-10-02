import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import (
    roc_auc_score, auc, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt  # matplotlib 추가
from dataloader import get_data_loaders  # 데이터 로더 함수 가져오기
from models import MLP  # MLP만 가져오기

# 랜덤 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 멀티모달 모델이 아닌, MLP만을 사용하는 모델 클래스 정의
class MLPModel(nn.Module):  # nn.Module 상속받음
    def __init__(self, mlp, mlp_feature_size):
        super(MLPModel, self).__init__()  # nn.Linear, nn.Conv2d 같은 모듈을 사용하도록 초기화
        self.mlp = mlp  # 임상 데이터를 처리할 MLP 네트워크
        self.fc = nn.Linear(mlp_feature_size, 1)  # 임상 데이터 feature 벡터로 최종 이진 분류
        self.sigmoid = nn.Sigmoid()  # 이진 분류용 시그모이드 활성화 함수
    
    def forward(self, clinical_data):  # 순전파 과정
        clinical_out, clinical_features = self.mlp(clinical_data)  # MLP를 통해 임상 데이터 특징 추출
        out = self.fc(clinical_features)  # 특징 벡터로 최종 출력 생성
        out = self.sigmoid(out)  # 시그모이드 활성화 함수 적용
        return out.squeeze(1)  # 출력에서 불필요한 차원 제거

# 모델을 학습하는 함수
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, patience=5, save_path='/home/alleun1110/Digital_Bio/MLP_CNN/best_model_mlp.pth'): 
    best_valid_loss = float('inf')  # 초기 검증 손실 값 설정
    best_valid_auc = 0.0  # 최적의 AUC 저장
    epochs_no_improve = 0  # 성능 개선이 없었던 에포크 수
    
    for epoch in range(num_epochs):
        model.train()  # 학습 모드로 설정
        train_loss = 0.0
        
        for _, clinical_data, labels in train_loader:
            clinical_data = clinical_data.cuda()  # GPU로 이동
            labels = labels.float().cuda()  # 레이블을 float 타입으로 변환 후 GPU로 이동

            optimizer.zero_grad()  # 기울기 초기화
            outputs = model(clinical_data)  # 모델에 데이터를 입력하여 예측값 얻기
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 옵티마이저로 가중치 업데이트
            train_loss += loss.item() * clinical_data.size(0)  # 배치 손실 계산

        train_loss = train_loss / len(train_loader.dataset)  # 평균 학습 손실 계산

        scheduler.step()  # 학습률 스케줄러 갱신

        # 검증 단계
        model.eval()  # 평가 모드 설정
        valid_loss = 0.0
        valid_labels = []
        valid_outputs = []
        
        with torch.no_grad():
            for _, clinical_data, labels in valid_loader:
                clinical_data = clinical_data.cuda()
                labels = labels.float().cuda()

                outputs = model(clinical_data)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * clinical_data.size(0)

                valid_labels.extend(labels.cpu().numpy())
                valid_outputs.extend(outputs.cpu().numpy())

        valid_loss = valid_loss / len(valid_loader.dataset)  # 평균 검증 손실 계산
        valid_auc = roc_auc_score(valid_labels, valid_outputs)  # AUC 계산

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}')

        if valid_loss < best_valid_loss:  # 성능 개선 여부 확인
            best_valid_loss = valid_loss
            best_valid_auc = valid_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)  # 최적 모델 저장
            print(f"Best model saved at epoch {epoch+1} to {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:  # Early Stopping
            print(f'Early stopping at epoch {epoch+1}')
            break

# 모델 평가 함수
def evaluate_model(model, test_loader, save_path='/home/alleun1110/Digital_Bio/MLP_CNN/auroc_curve_mlp.png'):
    model.eval()  # 평가 모드 설정
    test_labels = []
    test_outputs = []
    
    with torch.no_grad():  # 기울기 계산 없이 평가
        for _, clinical_data, labels in test_loader:
            clinical_data = clinical_data.cuda()
            labels = labels.float().cuda()

            outputs = model(clinical_data)

            test_labels.extend(labels.cpu().numpy())
            test_outputs.extend(outputs.cpu().numpy())

    test_labels = np.array(test_labels)
    test_outputs = np.array(test_outputs)
    test_outputs_binary = (test_outputs > 0.5).astype(int)

    # 메트릭 계산
    auc_score = roc_auc_score(test_labels, test_outputs)
    accuracy = accuracy_score(test_labels, test_outputs_binary)
    precision = precision_score(test_labels, test_outputs_binary)
    recall = recall_score(test_labels, test_outputs_binary)
    f1 = f1_score(test_labels, test_outputs_binary)
    cm = confusion_matrix(test_labels, test_outputs_binary)

    print(f'Test AUC: {auc_score:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{cm}')

    # ROC Curve 그리기 및 저장
    fpr, tpr, _ = roc_curve(test_labels, test_outputs)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # PNG 파일로 ROC curve 저장
    plt.savefig(save_path)
    print(f'ROC curve saved at {save_path}')
    plt.close()

# main 함수: MLP 모델만 학습 및 평가
def main():
    # 랜덤 시드 고정
    set_seed(42)
    
    train_loader, valid_loader, test_loader = get_data_loaders('/home/alleun1110/Digital_Bio/MLP_CNN/annotations.csv')

    clinical_input_size = train_loader.dataset[0][1].shape[0]  # 임상 데이터 입력 크기
    mlp = MLP(input_size=clinical_input_size).cuda()  # MLP 모델 초기화 및 GPU로 이동

    mlp_feature_size = 64  # MLP의 최종 출력 특징 벡터 크기

    model = MLPModel(mlp, mlp_feature_size).cuda()  # MLP 모델 생성 및 GPU로 이동
    criterion = nn.BCELoss()  # 이진 크로스 엔트로피 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 매 10 에포크마다 학습률을 0.1배로 줄임

    num_epochs = 100
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs)

    # 테스트 데이터에 대해 모델 평가 및 AUROC 커브 저장
    evaluate_model(model, test_loader, save_path='/home/alleun1110/Digital_Bio/MLP_CNN/auroc_curve_mlp.png')

if __name__ == '__main__':
    main()  # main 함수 실행

