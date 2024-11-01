import torch
import torch.nn.functional as F
from torch import nn
from OT_loss import OTLoss
from .cross_entropy_loss import CrossEntropyLoss

class OT_Distillation(CrossEntropyLoss):
    def __init__(self, args, student_dim, teacher_dim, padding_id=-100, reg=0.1) -> None:
        super().__init__(args, padding_id=padding_id)

        self.ot_loss = OTLoss(args= args, input_dim=student_dim, output_dim=teacher_dim,sinkhorn_epsilon=reg)


    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )

        hidden_state_student = outputs.hidden_states[-1]  # (batch_size, seq_len_student, hidden_dim_student)
        logits = outputs.logits
        log = {}
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
            
            hidden_state_teacher = teacher_outputs.hidden_states[-1]  # (batch_size, seq_len_teacher, hidden_dim_teacher)

       
        ot_loss = self.ot_loss(hidden_state_teacher, hidden_state_student)
        log["ot_loss"] = ot_loss.item()

        total_loss = loss + ot_loss

        # Compute accuracy for logging
        accuracy = self.compute_token_accuracy(logits, output_data["label"])
        log["accuracy"] = accuracy
        log["ot_loss"] = ot_loss.item()
        log["total_loss"] = total_loss.item()

        if self.args.report_logits:
            self.record_logits(
                logits, 
                output_data["label"], 
                log, 
                teacher_target=output_data[f"teacher_{distiller.teacher_model_type}_label"]
            )

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return total_loss / batch_denom, logging_output
