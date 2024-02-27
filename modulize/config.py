import torch
import torch.nn.functional as F
import torchmetrics
from modulize.model import *

config = {
    'ResNet' : {
        'model' : ResNet50(),
        
        'train_params' : {
            'data_loader_params' : 
                {
                'batch_size' : 32,
                'max_images_per_class' : 3000,
            
            },
            'epochs' : 20,
            'loss' : F.cross_entropy,
            'optim' : torch.optim.AdamW,
            'optim_params' : {
                'lr' : 0.0001,
                'name':'AdamW',
            },
            'model_save_path':"model/ResNetmodel.pth",
            'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'epochs' : 50, # Multi 300
        },
    },
    
    'VitClassification':{
        'model' : ViTForEmotionClassification(),
        
        'train_params' : {
            'data_loader_params' : {
            'batch_size' : 16,
            'max_images_per_class' : 3000,
            
        },
        'epochs' : 20,
        'loss' : F.cross_entropy,
        'optim' : torch.optim.AdamW,
        'optim_params' : {
            'lr' : 0.0001,
            'name':'AdamW',
        },
        
        'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'epochs' : 50, # Multi 300
        },
    }
    ,
    'ResNetDetection':{
        'model' : ResNetMultitask(7,4),
        
        'train_params' : {
            'data_loader_params' : {
                'batch_size' : 16,
                'max_images_per_class' : 3000,
            
            },
            'epochs' : 20,
            'loss' : "l1",  #put 'l1' or 'iou' here
            'optim' : torch.optim.AdamW,
            'optim_params' : {
                'lr' : 0.0001,
                'name':'AdamW',
            },
        
            'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'epochs' : 50, # Multi 300
            },
    }
    
    ,
    'VitDetection':{
        'model' : ViTMultitask(7,4),
        
        'train_params' : {
            'data_loader_params' : {
                'batch_size' : 16,
                'max_images_per_class' : 3000,
            
            },
            'epochs' : 20,
            'loss' : "l1",  #put 'l1' or 'iou' here
            'optim' : torch.optim.AdamW,
            'optim_params' : {
                'lr' : 0.0001,
                'name':'AdamW',
            },
        
            'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'epochs' : 50, # Multi 300
            },
    }
}