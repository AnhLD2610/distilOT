import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
import numpy as np

def min_max_scaling(C):
    """Min-max scaling for stabilization"""
    eps = 1e-10
    if isinstance(C, torch.Tensor):
        C_min = torch.min(C)
        C_max = torch.max(C)
    else:
        C_min = np.min(C)
        C_max = np.max(C)
    C = (C - C_min + eps) / (C_max - C_min + eps)
    return C

def compute_distance_matrix_cosine(x1, x2):
    """Compute cosine distance matrix between two sets of vectors"""
    # Normalize the vectors
    x1_norm = F.normalize(x1, p=2, dim=-1)
    x2_norm = F.normalize(x2, p=2, dim=-1)
    
    # Compute cosine similarity matrix
    C = torch.matmul(x1_norm, x2_norm.t())
    
    # Convert to distance in range [0, 1]
    C = (C + 1.0) / 2.0
    C = 1.0 - C
    return C

def compute_distance_matrix_l2(x1, x2):
    """Compute L2 distance matrix between two sets of vectors"""
    C = torch.cdist(x1, x2, p=2)
    C = min_max_scaling(C)
    return C

def compute_weights_uniform(x1, x2):
    """Compute uniform weights for the vectors"""
    n1 = x1.size(0)
    print(n1)
    n2 = x2.size(0)
    print(n2)

    weights1 = torch.ones(n1, device=x1.device) / n1
    weights2 = torch.ones(n2, device=x2.device) / n2
    return weights1, weights2

def compute_weights_norm(s1_word_embeddigs, s2_word_embeddigs):
    s1_weights = torch.norm(s1_word_embeddigs, dim=1)
    s2_weights = torch.norm(s2_word_embeddigs, dim=1)
    return s1_weights, s2_weights

class OTLoss(nn.Module):
    def __init__(self, 
                 args,
                 input_dim=768,
                 output_dim=1600,
                 distance_type='cosine',
                 weight_type='uniform',
                 sinkhorn_epsilon=0.1,
                 sinkhorn_max_iter=100,
                 sinkhorn_threshold=1e-7):
        super().__init__(args)
        
        # Dimension reduction 

        # model-type is student 
        # teacher-model-type is teacher 
        if self.args.model_type == 'gpt2':
            input_dim = 768
        elif self.args.model_type == 'tinyllama':
            input_dim = 2048 

        if self.args.teacher_model_type == 'gpt2':
            output_dim = 1600
        elif self.args.teacher_model_type == 'qwen':
            output_dim = 2048
        elif self.args.teacher_model_type == 'llama2':
            output_dim = 4096
        elif self.args.teacher_model_type == 'mistral':
            output_dim == 4096

        self.vec_transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            #nn.ReLU()
            # nn.Tanh()
        )
        
        # OT parameters
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.sinkhorn_threshold = sinkhorn_threshold
        
        # Distance function selection
        self.distance_type = distance_type
        self.dist_func = (compute_distance_matrix_cosine 
                         if distance_type == 'cosine' 
                         else compute_distance_matrix_l2)
        
        # Weight function selection
        self.weight_type = weight_type
        self.weight_func = (compute_weights_norm 
                           if weight_type == 'norm' 
                           else compute_weights_uniform)
        
    def forward(self, teacher_outputs, student_outputs):
        """
        Compute OT loss between teacher and student outputs
        
        Args:
            teacher_outputs: tensor of shape (batch_size, seq_len1, input_dim)
            student_outputs: tensor of shape (batch_size, seq_len2, output_dim)
            
        Returns:
            loss: scalar tensor
        """
        batch_size = teacher_outputs.size(0)
        
        # Transform students outputs to match teachers dimension
        student_outputs = self.vec_transform(student_outputs)
        
        total_loss = 0
        
        for b in range(batch_size):
            # Get sequences for current batch
            teacher_seq = teacher_outputs[b]
            student_seq = student_outputs[b]
            
            # Compute cost matrix
            C = self.dist_func(teacher_seq, student_seq)
            
            # Compute weights
            weights1, weights2 = self.weight_func(teacher_seq, student_seq)
            
            # Ensure weights sum to 1
            weights1 = weights1 / weights1.sum()
            weights2 = weights2 / weights2.sum()
            
            # Convert to numpy if using POT backend
            if not isinstance(C, np.ndarray):
                C = C.detach().cpu().numpy()
                weights1 = weights1.detach().cpu().numpy()
                weights2 = weights2.detach().cpu().numpy()
            
            # Compute OT matrix
            P = ot.sinkhorn(
                weights1, 
                weights2, 
                C,
                reg=self.sinkhorn_epsilon,
                numItermax=self.sinkhorn_max_iter,
                stopThr=self.sinkhorn_threshold
            )
            
            # Convert back to tensor if necessary
            if isinstance(C, np.ndarray):
                P = torch.from_numpy(P).to(teacher_outputs.device)
                C = torch.from_numpy(C).to(teacher_outputs.device)
            
            # Compute loss for current batch
            batch_loss = torch.sum(P * C)
            total_loss += batch_loss
            
        return total_loss / batch_size

# # Example usage
# def test_ot_loss():
#     # Create dummy data
#     batch_size = 2
#     teacher_seq_len = 10
#     student_seq_len = 8
#     teacher_dim = 4096
#     student_dim = 768
    
#     teacher_outputs = torch.randn(batch_size, teacher_seq_len, teacher_dim)
#     student_outputs = torch.randn(batch_size, student_seq_len, student_dim)
    
#     # Initialize loss function
#     ot_loss = OTLoss(
#         input_dim=teacher_dim,
#         output_dim=student_dim,
#         distance_type='l2',
#         weight_type='norm_uniform',
#         sinkhorn_epsilon=0.1
#     )
    
#     # Compute loss
#     loss = ot_loss(teacher_outputs, student_outputs)
#     print(f"OT Loss: {loss.item()}")
    
# if __name__ == "__main__":
#     test_ot_loss()