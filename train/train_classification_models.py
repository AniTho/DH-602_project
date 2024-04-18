import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from time import time
from prettytable import PrettyTable
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
  
def train_classification_model(cfg, train_dataset, valid_dataset, test_dataset):
    
    model_name = cfg.MODEL.BACKBONE
    class_names = ['Healthy', 'Doubtful', 'Minimal', 'Moderate', 'Severe']
    epochs = cfg.TRAIN.NUM_EPOCHS
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TRAIN.NUM_WORKERS
    save_path = f'{cfg.TRAIN.MODEL_PATH}/{model_name}_{cfg.DATA.LONG_TAILED}.pth'
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=cfg.TRAIN.PIN_MEMORY)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=cfg.TRAIN.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=cfg.TRAIN.PIN_MEMORY)
    
    kl_grades = []
    for batch in train_loader:
        kl_grade = batch['kl_grade']
        kl_grades.extend(kl_grade)
    
    classes, counts = np.unique(kl_grades, return_counts=True)
    print("Training Samples: ", dict(zip(classes, counts)))
    print(f"Classes: {classes}")
    print(f"Class Names: {class_names}")

    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.TRAIN.LR))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=cfg.TRAIN.PATIENCE, factor=cfg.TRAIN.FACTOR, min_lr=float(cfg.TRAIN.MIN_LR))

    def train_model():
        model.train()
        best_loss = float('inf')
        start_time = time()
        for epoch in range(epochs):
            running_loss = 0.0
            loop = tqdm(train_loader, total=len(train_loader))
            for batch in loop:
                inputs, labels = batch['image'], batch['kl_grade']
                if inputs.size(1) == 1:  
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
                loop.set_postfix(loss=running_loss / len(train_dataset))

            train_loss = running_loss / len(train_dataset)
            val_loss, val_class_accuracies = evaluate(valid_loader, log_classwise=True)
            scheduler.step(val_loss)

            # Log losses and accuracies to wandb
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, **val_class_accuracies, "epoch": epoch+1})

            # Save the model if validation loss has decreased
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), save_path)

        time_elapsed = time() - start_time
        print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    
    def evaluate(loader, log_classwise=False):
        model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        all_labels = []
        all_predictions = []
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch['image'], batch['kl_grade']
                if inputs.size(1) == 1:  
                    inputs = inputs.repeat(1, 3, 1, 1)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        classwise_accuracy = {f'val_acc_kl_grade_{classes[i]}': 100 * class_correct[i] / class_total[i] 
                            for i in range(len(classes)) if class_total[i] > 0}
        
        overall_accuracy = 100 * correct / total

        if log_classwise:
            # Log classwise accuracies during validation only
            wandb.log(classwise_accuracy)
            wandb.log({"val_overall_acc": overall_accuracy})

        if loader == test_loader:
            # Print classwise accuracy table at the end when using test_loader
            table = PrettyTable(["Class", "Correct", "Total", "Classwise Accuracy"])
            test_classwise_acc = {f'test_acc_kl_grade_{classes[i]}': 100 * class_correct[i] / class_total[i] 
                                for i in range(len(classes)) if class_total[i] > 0}
            wandb.log(test_classwise_acc)
            for i in range(len(classes)):
                table.add_row([classes[i], int(class_correct[i]), int(class_total[i]),
                            100 * class_correct[i] / class_total[i]])
            print(table)
            
            print(f'Test Overall Accuracy: {overall_accuracy:.2f}%')
            wandb.log({"test_overall_acc": overall_accuracy})
            
            # Generate and save confusion matrix
            cm = confusion_matrix(all_labels, all_predictions)
            print(f"Confusion Matrix: \n{cm}")
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion_Matrix_{model_name}_{cfg.DATA.LONG_TAILED}')
            plt.tight_layout()              
            cm_path = f'./plots/confusion_matrix_{model_name}_{cfg.DATA.LONG_TAILED}.png'
            plt.savefig(cm_path)
            print(f'Confusion Matrix exported to: {cm_path}')
            plt.close()

            # Upload confusion matrix to wandb
            wandb.log({"confusion_matrix": wandb.Image(cm_path)})

            # Generate and save classification report
            report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
            print(report)

        return total_loss / len(loader.dataset), classwise_accuracy

    # Execution starts here
    print('Training Model...')
    train_model()

    print('Loading Best Model from', save_path)
    model.load_state_dict(torch.load(save_path))

    print('Testing Model...')
    evaluate(test_loader, log_classwise=False)

