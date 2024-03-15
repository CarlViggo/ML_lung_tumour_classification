from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from partially_pretrained_architecture import gen_alexnet, simple_alexnet
from utils import get_train_valid_loader, get_test_loader, get_loader_for_fold
import matplotlib.pyplot as plt

def main_kfold():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'datasets/pul_nodules_entire'
    num_classes = 2
    num_epochs = 500
    batch_size = 128
    learning_rate = 0.001
    image_size = 128
    k_folds = 10
    augment = True
    imagenet = False

    seed = 69

    patience = 30 
    best_val_loss = float('inf')
    current_patience = 0

    dataset = datasets.ImageFolder(root='datasets/pul_nodules')

    # K-Fold Cross-Validation setup
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Cross-validation loop
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Setup KFold
    
        # Adjust indices dictionary to match get_loader_for_fold requirements
        indices = {'train': train_ids, 'valid': test_ids}

        # Use the custom loader function for the current fold
        train_loader, valid_loader = get_loader_for_fold(data_dir=data_dir, 
                                                        indices=indices, 
                                                        batch_size=batch_size, 
                                                        augment=augment, 
                                                        size=image_size, 
                                                        imagenet=imagenet)

        # Initialize your model, loss criterion, optimizer, etc. here for each fold
        model = gen_alexnet(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=8)

        total_step_train = len(train_loader)

        validation_accuracies = []
        testing_accuracies = []
        training_accuracies = []
        training_losses = []
        validation_losses = []

        best_val_loss = float('inf')
        current_patience = 0

        for epoch in range(num_epochs):
            model.train() 
            loss_sum = 0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()

                # training labels 
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_acc = correct / total * 100
            training_accuracies.append(train_acc)

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} Accuracy: {:.4f}%' 
                        .format(epoch+1, num_epochs, i+1, total_step_train, loss_sum/total_step_train, train_acc))
            
            #save losses
            average_train_loss = loss_sum / total_step_train
            training_losses.append(average_train_loss)

            # Validation
            model.eval() 
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)

                    # calculate validation loss to update lr schedule 
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_accuracy = 100 * correct / total
                print('Accuracy of the network on the {} validation images: {:.4f} %'.format(total, val_accuracy))
                print(" ")
                validation_accuracies.append(val_accuracy)
                average_val_loss = val_loss / len(valid_loader)
                validation_losses.append(average_val_loss)

                # check if validation loss has improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    current_patience = 0
                    best_model_dict = model.state_dict()
                else:
                    current_patience += 1

                # check for early stopping
                if current_patience >= patience:
                    print(f'Early stopping after {epoch} epochs.')
                    break

                # update lr schedule 
                scheduler.step(val_loss / len(valid_loader))

                if epoch % 5 == 0: 
                    # Save model that's noe compatible with biggan-am,
                    torch.save(best_model_dict, f'fold_{fold}_pul_nod_model_ep{epoch}_DICT.pth')
                    torch.save(model, f'fold_{fold}_pul_nod_model_ep{epoch}_DICT.pth')
            
            # Save model that's noe compatible with biggan-am,
            torch.save(best_model_dict, f'fold_{fold}_pul_nod_model_FINAL_DICT.pth')
            torch.save(model, f'fold_{fold}_pul_nod_model_FINAL_DICT.pth')

            torch.save({
            'training_accuracies': training_accuracies, 
            'validation_accuracies': validation_accuracies,
            'training_losses': training_losses, 
            'validation_losses': validation_losses
            }, f'fold_{fold}_model_metrics.pth')

   

if __name__ == '__main__':
    main_kfold()
