import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
from torchmetrics.classification import BinaryConfusionMatrix
import torch.optim as optim
from Model import Binary_Classifier
import os
import logging
import plot
import numpy as np

def setup_dataset(traindir:str, validationdir:str, testdir:str=None, dim=224, batch: int=16) -> tuple:
    class_names = sorted(os.listdir(traindir))
    # Assert that class names are names in this scheme: 00_class, 01_class, ...
    for c in class_names:
        assert len(
            c.split("_")[0]) == 2, "Class names must start with double digit numbers followed by an underscore"
    #transformations
    img_transforms = transforms.Compose([transforms.Resize((dim,dim)),
                                        transforms.ToTensor(),                                
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])
                            
    #datasets
    train_data = datasets.ImageFolder(os.path.normpath(traindir),transform=img_transforms)
    valid_data = datasets.ImageFolder(os.path.normpath(validationdir),transform=img_transforms)
    if testdir:
        test_data = datasets.ImageFolder(os.path.normpath(testdir),transform=img_transforms) 
    else:
        test_data = valid_data

    #dataloader
    trainset = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=batch)
    validset = torch.utils.data.DataLoader(valid_data, shuffle = True, batch_size=batch)
    testset = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=batch)

    logging.info("Successfully loaded dataset")

    return trainset, validset, testset, class_names


def train(trainset, m_device, m_model, m_optimizer, m_loss, m_epoch:int, train_loss_values:list,train_acc_values:list, log_interval=200)->None:
    # Set model to training mode
    m_model.train()
    train_loss,correct = 0,0

    for batch_idx, (X_train, y_train) in enumerate(trainset):


        X_train = X_train.to(m_device)
        y_train = y_train.to(m_device)

        # Forward Propagation:  compute predicted outputs by passing inputs to the model
        y_predicted = m_model(X_train)
        # Zero gradient buffers: clear the gradients of all optimized variables
        m_optimizer.zero_grad() 

        # Calculate loss
        loss = m_loss(y_predicted, y_train.float().unsqueeze(1))

        # Backpropagate: compute gradient of the loss with respect to m_model parameters
        loss.backward()
        
        # Update weights: perform a single optimization step (parameter update)
        m_optimizer.step()
        train_loss += loss.item()  
        pred = y_predicted.data.max(1)[1]
        correct += (pred == y_train).float().sum()
 
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                m_epoch, batch_idx * len(X_train), len(trainset.dataset),
                100. * batch_idx / len(trainset), loss.item()))
    
    # calculate average loss over an epoch            
    train_loss /= len(trainset)
    train_loss_values.append(train_loss)    
    
    accuracy = 100 * correct / len(trainset.dataset)
    train_acc_values.append(accuracy.item())

    print(f'Train Epoch: {m_epoch}\nTrain set: Average loss: {loss}, Accuracy: {correct}/{len(trainset.dataset)} --> {round(accuracy.item(),2)}%\n')
    print(f"train_acc: {train_acc_values}, lenght: {len(train_acc_values)}")
    print(f"train_loss: {train_loss_values}, lenght: {len(train_loss_values)}\n")

def validate(valset, m_device, m_model,m_loss, m_epoch, val_loss_values:list, val_acc_values:list)->None:
    m_model.eval()
    val_loss, correct = 0, 0
    
    for X_val, y_val in valset:

        X_val = X_val.to(m_device)
        y_val = y_val.to(m_device)

        # Forward Propagation:  compute predicted outputs by passing inputs to the model
        y_predicted = m_model(X_val)
        
        # Calculate loss
        loss = m_loss(y_predicted, y_val.float().unsqueeze(1))
        val_loss += loss.item()  
        
        pred = y_predicted.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(y_val.data).cpu().sum()

    val_loss /= len(valset)
    val_loss_values.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(valset.dataset)
    val_acc_values.append(accuracy)
    
    print(f'Val Epoch: {m_epoch}\nValidation set: Average loss: {loss}, Accuracy: {correct}/{len(valset.dataset)} --> {round(accuracy.item(),2)}%\n')
    print(f"val_acc: {val_acc_values}, lenght: {len(val_acc_values)}")
    print(f"val_loss: {val_loss_values}, lenght: {len(val_loss_values)}\n") 

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%y %H:%M:%S',
                            level=logging.DEBUG)
                            
    # kaggle kernels output gauravduttakiit/metal-defect-classification-using-mobilenet-v2 -p /path/to/dest
    traindir = "../casting_data/train"
    validdir = "../casting_data/valid"
    testdir = "../casting_512x512" # if None, testset = validset
 
    # Hyperparameters
    param_args = {'num_classes': 2,    
                    'Batch_Size': 8,
                    'dim':24, # dim*dim matrix for resizing method
                    'Learning_Rate': 0.01,       
                    'Epochs': 15 } 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Setup Dataset")
    train_loader, valid_loader, test_loader, label = setup_dataset(traindir, validdir, testdir, param_args['dim'],param_args['Batch_Size'])

    model = Binary_Classifier(input=3).to(device)
    # model = torch.load("./binary_model.pth")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=param_args['Learning_Rate']) 
    loss = nn.BCELoss() # BinaryCrossEntropy Loss #nn.BCEWithLogitsLoss
    train_loss,val_loss, train_acc, val_acc = list(), list(), list(), list()
    start = time.time()
    logging.info("Starting training...")
    for epoch in range(param_args['Epochs']):
        train(trainset=train_loader, m_device= device, m_model=model, 
                m_optimizer=optimizer, m_loss=loss, m_epoch=epoch,
                train_loss_values=train_loss,train_acc_values=train_acc)
        validate(valset=valid_loader, m_device=device, m_model=model,
                    m_loss=loss, m_epoch=epoch,
                    val_loss_values=val_loss,val_acc_values=val_acc)
    elapsed_time = start - time.time()
    logging.info("End of Training")
    plot.save_model(model)
    print(" ")
    plot.save_graph(param_args['Epochs'], train_loss, val_loss, title='Training-loss')
    plot.save_graph(param_args['Epochs'], train_acc, val_acc, title='Training-accuracy')

    hours = elapsed_time // 3600
    rest = elapsed_time % hours
    minutes = rest // 60
    seconds = round(rest % 60)
    logging.info("training duration: {} sec -> {}h:{}min:{}sec".format(elapsed_time, int(hours),int(minutes),int(seconds)))

    # confusion matrix
    bcm = BinaryConfusionMatrix()
    model.eval()
    with torch.no_grad():
        for xb_test,yb_test  in test_loader:
            y_test_pred = model(xb_test.to(device))
            y_pred_tag = torch.round(y_test_pred).squeeze()
            bcm.update(y_pred_tag, yb_test)

    cf_matrix = bcm.compute().detach().numpy()
    print(f"Confusion matrix {cf_matrix}")
    plot.save_confusion_matrix(cf_matrix, label)
    

    
