import torch
from antihallucination.core import HallucinationMitigator

if __name__ == "__main__":
    # 가상의 LLM 출력 데이터 생성 (Batch=4, Seq_Len=16, Vocab_Dim=32000, Hidden_Dim=4096)
    dummy_logits = torch.randn(4, 16, 32000)
    dummy_hidden = torch.randn(4, 16, 4096)
    dummy_context = torch.randn(4, 10, 4096) # 정답 정보 임베딩
    
    # 환각 제거 엔진 초기화
    mitigator = HallucinationMitigator(temperature=0.7, beta_entropy=0.15)
    
    # 엔진 가동 및 오차 추출
    loss_total, loss_ent, loss_proj = mitigator(dummy_logits, dummy_hidden, dummy_context)
    
    print("=== Anti-Hallucination Engine Integrity Check ===")
    print(f"1. Total Mitigation Loss: {loss_total.item():.4f}")
    print(f"2. Token Uncertainty Entropy: {loss_ent.item():.4f}")
    print(f"3. Geometric Context Distortion: {loss_proj.item():.4f}")
