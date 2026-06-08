import torch
import torch.nn as nn
import torch.nn.functional as F

class HallucinationMitigator(nn.Module):
    def __init__(self, temperature=1.0, beta_entropy=0.1):
        super(HallucinationMitigator, self).__init__()
        self.temperature = temperature
        self.beta_entropy = beta_entropy
        
        # 가량 학습 과정에서의 가중치 보정 값 (In-place 충돌 방지를 위해 float 버퍼로 관리)
        self.alpha_scale = 1.0

    def compute_context_entropy(self, logits):
        """언어 모델 출력 확률(Logits)의 섀넌 엔트로피를 계산하여 불확실성을 측정합니다."""
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return torch.mean(entropy)

    def contextual_projection_loss(self, hidden_states, context_embeddings):
        """숨겨진 상태(Hidden States)를 컨텍스트 매니폴드 공간으로 투영하여 정보 기하학적 왜곡을 계산합니다."""
        # Pairwise Euclidean Distance 계산 (Batch x Seq_len x Context_len)
        dist_matrix = torch.cdist(hidden_states, context_embeddings, p=2)
        
        # 최소 거리 기반의 기하학적 정렬 오차 추출
        min_distances = torch.min(dist_matrix, dim=-1)[0]
        return torch.mean(min_distances)

    def forward(self, logits, hidden_states, context_embeddings):
        """환각 확률과 컨텍스트 위상 왜곡을 결합하여 무결성 손실 함수를 도출합니다."""
        L_entropy = self.compute_context_entropy(logits)
        L_projection = self.contextual_projection_loss(hidden_states, context_embeddings)
        
        # 두 다중 목적 오차를 결합하여 최종 환각 억제 오차 도출
        L_total = L_entropy + self.alpha_scale * L_projection
        
        return L_total, L_entropy, L_projection
