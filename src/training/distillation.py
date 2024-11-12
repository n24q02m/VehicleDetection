from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
import torch.nn.functional as F
import torch


class DistillationTrainer(DetectionTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load teacher model
        self.teacher_model = YOLO("./runs/finetune_yolov8m/weights/best.pt")
        self.teacher_model.model.to(self.device)
        self.teacher_model.model.eval()  # Set teacher model to evaluation mode
        for param in self.teacher_model.model.parameters():
            param.requires_grad = False  # Freeze teacher model parameters

    def compute_loss(self, preds, batch):
        # Compute the standard loss
        loss, loss_items = self.loss(preds, batch)

        # Get student outputs
        student_logits = preds[0]

        # Get teacher outputs
        with torch.no_grad():
            teacher_preds = self.teacher_model.model(batch["img"])
            teacher_logits = teacher_preds[0]

        # Compute knowledge distillation loss
        kd_loss = self.compute_kd_loss(student_logits, teacher_logits)

        # Combine losses
        total_loss = loss + kd_loss
        loss_items["kd_loss"] = kd_loss.item()

        return total_loss, loss_items

    def compute_kd_loss(
        self, student_logits, teacher_logits, temperature=5.0, alpha=0.5
    ):
        """
        Compute the knowledge distillation loss using Kullback-Leibler divergence.
        """
        # Soft targets
        p_teacher = F.softmax(teacher_logits / temperature, dim=1)
        p_student = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (
            temperature**2
        )

        # Scale the loss
        kd_loss = alpha * kd_loss
        return kd_loss
