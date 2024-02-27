import torch 
import torch.nn as nn

def iou_loss(bbox_preds, bbox_labels):
    # Calculate intersection
    xA = torch.max(bbox_preds[:, 2], bbox_labels[:, 2])
    yA = torch.max(bbox_preds[:, 3], bbox_labels[:, 3])
    xB = torch.min(bbox_preds[:, 0], bbox_labels[:, 0])
    yB = torch.min(bbox_preds[:, 1], bbox_labels[:, 1])
    
    interArea = torch.clamp(xB - xA , min=0) * torch.clamp(yB - yA , min=0)
    
    # Calculate union
    boxAArea = (bbox_preds[:, 0] - bbox_preds[:, 2] ) * (bbox_preds[:, 1] - bbox_preds[:, 3] )
    boxBArea = (bbox_labels[:, 0] - bbox_labels[:, 2] ) * (bbox_labels[:, 1] - bbox_labels[:, 3] )
    unionArea = boxAArea + boxBArea - interArea
    
    # Compute the IoU
    iou = interArea / unionArea
    
    # IoU loss
    return 1 - iou.mean() 


def compute_loss_iou(class_logits, bbox_preds, class_labels, bbox_labels):
    cross_entropy_loss = nn.CrossEntropyLoss()  # Corrected usage: initialize loss function
    loss_class = cross_entropy_loss(class_logits, class_labels)
    #bbox_labels = torch.tensor(bbox_labels, dtype=torch.float, device=bbox_preds.device)
    loss_bbox = iou_loss(bbox_preds, bbox_labels)
    lambda_param = 3.0
    combined_loss = loss_class + lambda_param * loss_bbox
    return combined_loss, loss_class, loss_bbox




def smooth_l1_loss(bbox_preds, bbox_labels):
    l1_loss = nn.SmoothL1Loss()
    return l1_loss(bbox_preds, bbox_labels)

def compute_l1_loss(class_logits, bbox_preds, class_labels, bbox_labels):
    cross_entropy_loss = nn.CrossEntropyLoss()  # Corrected usage: initialize loss function
    loss_class = cross_entropy_loss(class_logits, class_labels)
    bbox_labels = torch.tensor(bbox_labels, dtype=torch.float, device=bbox_preds.device)
    loss_bbox = smooth_l1_loss(bbox_preds, bbox_labels)
    lambda_param = 3.0
    combined_loss = loss_class + lambda_param * loss_bbox
    return combined_loss, loss_class, loss_bbox