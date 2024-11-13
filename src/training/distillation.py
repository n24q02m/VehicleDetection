from copy import copy

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER

import torch
import torch.nn.functional as F

class DistillationTrainer(DetectionTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load teacher model
        self.teacher_model = YOLO("./runs/finetune_yolo11x/weights/best.pt")
        self.teacher_model.model.to(self.device)
        self.teacher_model.model.eval()  # Set teacher model to evaluation mode
        for param in self.teacher_model.model.parameters():
            param.requires_grad = False  # Freeze teacher model parameters
        # Override loss_names to include kd_loss
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "kd_loss")

    def get_validator(self):
        """Returns a DistillationValidator for validation with teacher model."""
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "kd_loss")
        validator = DistillationValidator(
            self.test_loader, 
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )
        validator.teacher_model = self.teacher_model  # Pass teacher model to validator
        return validator

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

        # Log the kd_loss value
        self.log_kd_loss(kd_loss.item())

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

    def log_kd_loss(self, kd_loss_value):
        """Log the knowledge distillation loss value."""
        if self.epoch == 0:
            self.metrics["kd_loss"] = []
        self.metrics["kd_loss"].append(kd_loss_value)
        LOGGER.info(f"Knowledge Distillation Loss: {kd_loss_value}")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names + ("kd_loss",)]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys


class DistillationValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "kd_loss")
        self.teacher_model = None  # Will be set by DistillationTrainer

    def compute_loss(self, preds, batch):
        """Compute loss including knowledge distillation loss during validation."""
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

    def compute_kd_loss(self, student_logits, teacher_logits, temperature=5.0, alpha=0.5):
        """Compute knowledge distillation loss."""
        p_teacher = F.softmax(teacher_logits / temperature, dim=1)
        p_student = F.log_softmax(student_logits / temperature, dim=1)
        kd_loss = F.kl_div(p_student, p_teacher, reduction="batchmean") * (temperature**2)
        return alpha * kd_loss
