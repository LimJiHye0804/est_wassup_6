import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from modulize.dataset import *
from modulize.model import *
from modulize.traineval import *
from modulize.loss import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from modulize.config import config
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, r2_score


def main(config, args):
    model_config = config[args.model]
    if not model_config:
        print(f"Model {args.model} is not supported.")
        return
    
    train_params = model_config['train_params']
    device = train_params['device']
    
    root_dirs = {
    'anger': '../data/anger',
    'anxiety': '../data/anxiety',
    'embarrass': '../data/embarrass',
    'happy': '../data/happy',
    'pain': '../data/pain',
    'sad': '../data/sad',
    'normal':'../data/normal'
    }
    
    root_dirs2 = {
    'anger': '../data/test_set1000/ang',
    'anxiety': '../data/test_set1000/anx',
    'embarrass': '../data/test_set1000/emb',
    'happy': '../data/test_set1000/hap',
    'pain': '../data/test_set1000/pai',
    'sad': '../data/test_set1000/sad',
    'normal':'../data/test_set1000/nor'
}
 

    
    if 'ResNet' == args.model:
        print("you are training ", args.model)
        
        train_dataset = ResNetDataset(root_dirs, train = True , max_images_per_class=train_params['data_loader_params']['max_images_per_class'])
        tst_dataset = ResNetDataset(root_dirs2, train = False, max_images_per_class=None)
        
        train_loader = DataLoader(train_dataset, batch_size = train_params['data_loader_params']['batch_size'], shuffle=True)
        test_loader = DataLoader(tst_dataset, batch_size=train_params['data_loader_params']['batch_size'], shuffle=False)
        
        model = model_config['model'].to(device)
        optimizer = train_params['optim'](model.parameters(), lr=train_params['optim_params']['lr'])
        criterion = nn.CrossEntropyLoss()
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
        print("device:", device)
        train_losses = []
        eval_losses = []
        for epoch in range(train_params['epochs']):
            train_loss, train_batch_losses = train(model, criterion, optimizer, train_loader, device)
            avg_loss, accuracy, precision, eval_batch_losses = evaluate(model, criterion, test_loader, device)
            scheduler.step(avg_loss)
            train_losses.extend(train_batch_losses)
            eval_losses.extend(eval_batch_losses)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}")
        
        model_save_path = train_params['model_save_path']
        torch.save(model, model_save_path)
        
        
    elif 'VitClassification' == args.model:
        print("you are training ", args.model)
        
        train_dataset = VitDataset(root_dirs, train = True, max_images_per_class=train_params['data_loader_params']['max_images_per_class'])
        tst_dataset = VitDataset(root_dirs2, train = False, max_images_per_class=None)
        
        train_loader = DataLoader(train_dataset, batch_size = train_params['data_loader_params']['batch_size'], shuffle=True)
        test_loader = DataLoader(tst_dataset, batch_size=train_params['data_loader_params']['batch_size'], shuffle=False)
        
        model = model_config['model'].to(device)
        optimizer = train_params['optim'](model.parameters(), lr=train_params['optim_params']['lr'])
        criterion = nn.CrossEntropyLoss()
        
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
        print("device:", device)

        train_losses = []
        eval_losses = []
        for epoch in range(train_params['epochs']):
            train_loss, train_batch_losses = train(model, criterion, optimizer, train_loader, device)
            avg_loss, accuracy, precision, eval_batch_losses = evaluate(model, criterion, test_loader, device)
            scheduler.step(avg_loss)
            train_losses.extend(train_batch_losses)
            eval_losses.extend(eval_batch_losses)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {avg_loss}, Accuracy: {accuracy}, Precision: {precision}")
        
        model_save_path = train_params['model_save_path']
        torch.save(model, model_save_path)
        
    
    elif 'VitDetection' == args.model:
        print("you are training ", args.model)
        
        train_info, test_info , transform , dict2 = make_dataset(vit=True)
        print("making dataset")
        train_dataset = VitDetectionDataset(train_info, dict2, transform= transform, max_images_per_class=train_params['data_loader_params']['max_images_per_class'])
        tst_dataset = VitDetectionDataset(test_info, dict2, transform= transform, max_images_per_class=1000)
        
        train_loader = DataLoader(train_dataset, batch_size = train_params['data_loader_params']['batch_size'], shuffle=True)
        test_loader = DataLoader(tst_dataset, batch_size=train_params['data_loader_params']['batch_size'], shuffle=False)

        model = model_config['model'].to(device)
        optimizer = train_params['optim'](model.parameters(), lr=train_params['optim_params']['lr'])
        
        if train_params['loss'] == "l1":
            criterion = compute_l1_loss
        else :
            criterion = compute_loss_iou
        
        model2, accuracy, precision, mse, r2 = detection_train_eval(train_params['epochs'], model, train_loader, test_loader, device, optimizer, criterion)
        
        model_save_path = train_params.model_save_path
        torch.save(model2, model_save_path)
        
    elif 'ResNetDetection' == args.model:
        print("you are training ", args.model)
        
        train_info, test_info , transform , dict2 = make_dataset(vit=False)
        print("making dataset")
        
        train_dataset = VitDetectionDataset(train_info, dict2, transform= transform, max_images_per_class=train_params['data_loader_params']['max_images_per_class'])
        tst_dataset = VitDetectionDataset(test_info, dict2, transform= transform, max_images_per_class=1000)
        
        train_loader = DataLoader(train_dataset, batch_size = train_params['data_loader_params']['batch_size'], shuffle=True)
        test_loader = DataLoader(tst_dataset, batch_size=train_params['data_loader_params']['batch_size'], shuffle=False)

        model = model_config['model'].to(device)
        optimizer = train_params['optim'](model.parameters(), lr=train_params['optim_params']['lr'])
        
        if train_params['loss'] == "l1":
            criterion = compute_l1_loss
        else :
            criterion = compute_loss_iou
        
        model2, accuracy, precision, mse, r2 = detection_train_eval(train_params['epochs'], model, train_loader, test_loader, device, optimizer, criterion)
        
        model_save_path = train_params.model_save_path
        torch.save(model2, model_save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face-Expression-Classification")
    parser.add_argument("-m", "--model", default="ResNet", type=str, help="Model to use")
    args = parser.parse_args()

    # Assuming the config is properly defined and imported
    main(config, args)