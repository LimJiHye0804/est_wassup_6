import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error, r2_score

def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    batch_losses = []  # To store loss every 10 batches
    for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 10 == 0:  # Collect loss every 10 batches
            batch_loss = running_loss / 10
            batch_losses.append(batch_loss)
            running_loss = 0.0  # Reset running loss
            tqdm.write(f'Batch {i+1}, Loss: {batch_loss}')
    avg_loss = sum(batch_losses) / len(batch_losses)
    return avg_loss, batch_losses

def evaluate(model, criterion, eval_loader, device):
    model.eval()
    running_loss = 0.0
    eval_batch_losses = []  # To store loss every 10 batches
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Collect loss every 10 batches
                batch_loss = running_loss / 10
                eval_batch_losses.append(batch_loss)
                running_loss = 0.0  # Reset running loss
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    
    print(sum(eval_batch_losses))
    print(len(eval_batch_losses))
    avg_loss = sum(eval_batch_losses) / len(eval_batch_losses)
    return avg_loss, accuracy, precision, eval_batch_losses


def detection_train_eval(epochs, model, train_loader, eval_loader, device, optimizer, criterion):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for images, class_labels, bbox_labels in progress_bar:
            images = images.to(device)
            class_labels = class_labels.to(device)
            
            if isinstance(bbox_labels, list):
                bbox_labels_flat = [coord for bbox in bbox_labels for coord in bbox]
                bbox_labels_tensor = torch.tensor(bbox_labels_flat, dtype=torch.float).view(-1, 4).to(device)
            else:
                bbox_labels_tensor = bbox_labels.to(device)
            
            optimizer.zero_grad()
            
            class_logits, bbox_preds = model(images)
            combined_loss, _, _ = criterion(class_logits, bbox_preds, class_labels, bbox_labels_tensor)
            
            combined_loss.backward()
            optimizer.step()
            
            train_loss += combined_loss.item()
            
            # Update the progress bar with the current batch loss
            progress_bar.set_postfix({'batch_loss': combined_loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}')
        
        # Evaluation phase
        model.eval()
        all_class_labels = []
        all_class_preds = []
        all_bbox_labels = []
        all_bbox_preds = []
        eval_loss = 0.0
        with torch.no_grad():
            for images2, class_labels2, bbox_labels2 in tqdm(eval_loader, desc='Evaluating', leave=False):
                images2 = images2.to(device)
                class_labels2 = class_labels2.to(device)
                if isinstance(bbox_labels2, list):
                    bbox_labels_flat2 = [coord2 for bbox2 in bbox_labels2 for coord2 in bbox2]
                    bbox_labels_tensor2 = torch.tensor(bbox_labels_flat2, dtype=torch.float).view(-1, 4).to(device)
                else:
                    bbox_labels_tensor2 = bbox_labels2.to(device)
                
                class_logits2, bbox_preds2 = model(images2)
                #print(bbox_labels_tensor2,bbox_preds2)
                class_preds = torch.argmax(class_logits2, dim=1)

                all_class_labels.extend(class_labels2.detach().cpu().numpy())
                all_class_preds.extend(class_preds.detach().cpu().numpy())
                all_bbox_labels.extend(bbox_labels_tensor2.detach().cpu().numpy())
                all_bbox_preds.extend(bbox_preds2.detach().cpu().numpy())
                
                combined_loss2, _, _ = criterion(class_logits2, bbox_preds2, class_labels2, bbox_labels_tensor2)
                
                eval_loss += combined_loss2.item()
        
        accuracy = accuracy_score(all_class_labels, all_class_preds)
        precision = precision_score(all_class_labels, all_class_preds, average='macro')
        
        # Bounding box regression metrics
        mse = mean_squared_error(all_bbox_labels, all_bbox_preds)
        r2 = r2_score(all_bbox_labels, all_bbox_preds)
        avg_eval_loss = eval_loss / len(eval_loader)
        print(f'Evaluation Loss: {avg_eval_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'MSE: {mse:.4f}')
        print(f'R2 Score: {r2:.4f}')
        
    return model, accuracy, precision, mse, r2