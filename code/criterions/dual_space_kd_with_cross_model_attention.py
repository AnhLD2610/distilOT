# import math
# import torch
# import torch.nn as nn
# from OT_loss import OTLoss

# class OT_Distillation(nn.Module):
#     def __init__(self, args, padding_id=-100) -> None:
#         super().__init__(args, padding_id=padding_id)
#         self.ot_loss = OTLoss()

#         def __init__(self, 
#                  input_dim=4096,
#                  output_dim=768,
#                  distance_type='cosine',
#                  weight_type='uniform',
#                  sinkhorn_epsilon=0.1,
#                  sinkhorn_max_iter=100,
#                  sinkhorn_threshold=1e-7):
    
#     def forward(
#         self, 
#         distiller, 
#         input_data, 
#         output_data, 
#         logging_output, 
#         batch_denom, 
#     ):
#         model = distiller.student_model
#         teacher_model = distiller.teacher_model
#         self.distiller = distiller
#         outputs = model(
#             input_data["input_ids"],
#             attention_mask=input_data["attention_mask"],
#             position_ids=input_data.get("position_ids", None), 
#             output_hidden_states=True
#         )
#         logits = outputs.logits
#         log = {}
#         loss = self.compute_cross_entropy_loss(
#             outputs.logits, output_data["label"], log=log
#         )[0]

#         with torch.no_grad():
#             teacher_model.eval()
#             teacher_outputs = teacher_model(
#                 input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
#                 attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
#                 position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
#                 output_hidden_states=True)
        

#          # Compute OT loss
#         ot_loss = self.ot_loss(teacher_outputs.hidden_states[-1], outputs.hidden_states[-1])
#         log["ot_loss"] = ot_loss.item()

#         loss = (1.0 - self.kd_rate) * loss + self.kd_rate * ot_loss
#         log["loss"] = loss

#         accuracy = self.compute_token_accuracy(
#             logits, output_data["label"], 
#         )
#         log["accuracy"] = accuracy

#         logging_output = self.record_logging_output(
#             logging_output, batch_denom, log
#         )
#         return loss / batch_denom, logging_output


# # import torch
# # import torch.nn as nn
# # import random

# # # # Define a sample OTLoss class (mock) for testing purposes
# # # class OTLoss(nn.Module):
# # #     def forward(self, teacher_hidden, student_hidden):
# # #         # Calculate a dummy OT loss as a simple L2 norm for testing
# # #         return torch.norm(teacher_hidden - student_hidden, p=2)

# # # Define a DualSpaceKDWithCMA class with just the loss calculation
# # class DualSpaceKDWithCMA(nn.Module):
# #     def __init__(self, kd_rate=0.5, padding_id=-100) -> None:
# #         super().__init__()
# #         self.kd_rate = kd_rate
# #         self.ot_loss = OTLoss()
    
# #     def compute_cross_entropy_loss(self, logits, labels, log=None):
# #         # Use CrossEntropyLoss as the base KD loss
# #         loss_fn = nn.CrossEntropyLoss(ignore_index=padding_id)
# #         loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
# #         if log is not None:
# #             log["ce_loss"] = loss.item()
# #         return loss

# #     def forward(self, student_logits, teacher_hidden, student_hidden, labels, batch_denom):
# #         log = {}
        
# #         # Compute Cross-Entropy Loss
# #         ce_loss = self.compute_cross_entropy_loss(student_logits, labels, log=log)
        
# #         # Compute OT Loss
# #         print(teacher_hidden.shape)
# #         print(student_hidden.shape)
# #         ot_loss = self.ot_loss(teacher_hidden, student_hidden)
# #         log["ot_loss"] = ot_loss.item()

# #         # Combine losses
# #         loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * ot_loss
# #         log["loss"] = loss.item()

# #         return loss / batch_denom, log

# # # Test setup
# # batch_size = 4
# # seq_len = 10
# # hidden_dim = 8
# # num_labels = 5
# # padding_id = -100
# # kd_rate = 0.5
# # batch_denom = batch_size

# # # Instantiate the KD loss model
# # model_kd = DualSpaceKDWithCMA(kd_rate=kd_rate, padding_id=padding_id)

# # # Generate random logits, hidden states, and labels for testing
# # student_logits = torch.randn(batch_size, seq_len, num_labels)  # Random logits for Cross-Entropy loss
# # teacher_hidden = torch.randn(batch_size, seq_len, 4096)  # Random hidden states for teacher
# # student_hidden = torch.randn(batch_size, seq_len, 768)  # Random hidden states for student
# # labels = torch.randint(0, num_labels, (batch_size, seq_len))   # Random labels

# # # Test the forward method to compute the loss
# # loss, log_output = model_kd(student_logits, teacher_hidden, student_hidden, labels, batch_denom)

# # # Print results
# # print("Loss:", loss)
# # print("Log Output:", log_output)
