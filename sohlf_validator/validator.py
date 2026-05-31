import numpy as np

class SOHLFValidator:
    def __init__(self, embedding_dim=128, bound_domain=0.05):
        self.embedding_dim = embedding_dim
        self.bound_domain = bound_domain

    def compute_so_hlf_index(self, ker_dim, coker_dim):
        return ker_dim - coker_dim

    def filter_hallucination(self, prompt_matrix, generated_token_vector):
        U, S, Vt = np.linalg.svd(prompt_matrix, full_matrices=False)
        pseudo_inverse_threshold = 1e-5
        is_ker = S > pseudo_inverse_threshold
        
        dim_ker = np.sum(is_ker)
        dim_coker = self.embedding_dim - dim_ker
        
        index_val = self.compute_so_hlf_index(dim_ker, dim_coker)
        coker_projection = np.eye(self.embedding_dim) - np.dot(U[:, is_ker], U[:, is_ker].T)
        hallucination_component = np.dot(coker_projection, generated_token_vector)
        
        b_so_hlf = np.linalg.norm(hallucination_component) - self.bound_domain
        
        if b_so_hlf <= 0:
            final_token_vector = generated_token_vector
            status = "VERIFIED"
        else:
            final_token_vector = generated_token_vector - hallucination_component
            status = "CORRECTED"
            
        return final_token_vector, status, index_val
