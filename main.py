import torch
import torch.nn as nn
import time
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from Model import Binary_Classifier
import os
import logging
from textwrap import wrap
import itertools
import json

def setup_dataset(traindir:str, testdir:str, batch: int) -> tuple:
    class_names = sorted(os.listdir(traindir))
    # Assert that class names are names in this scheme: 00_class, 01_class, ...
    for c in class_names:
        assert len(
            c.split("_")[0]) == 2, "Class names must start with double digit numbers followed by an underscore"
    #transformations
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),                                
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225],
        ),
                                        ])

    #datasets
    train_data = datasets.ImageFolder(traindir,transform=train_transforms)
    test_data = datasets.ImageFolder(testdir,transform=test_transforms)

    #dataloader
    trainset = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=batch)
    testset = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=batch)

    return trainset, testset, class_names


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
        loss = m_loss(y_predicted, y_train)

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
    train_acc_values.append(accuracy)
    
    print('Train Epoch: {}\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(m_epoch,
        loss, correct, len(trainset.dataset), accuracy))


def validate(valset, m_device, m_model,m_loss, m_epoch, val_loss_values:list, val_acc_values:list)->None:
    m_model.eval()
    val_loss, correct = 0, 0
    
    for X_val, y_val in valset:

        X_val = X_val.to(m_device)
        y_val = y_val.to(m_device)

        # Forward Propagation:  compute predicted outputs by passing inputs to the model
        y_predicted = m_model(X_val)
        
        # Calculate loss
        loss = m_loss(y_predicted, y_val)
        val_loss += loss.item()  
        
        pred = y_predicted.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(y_val.data).cpu().sum()

    val_loss /= len(valset)
    val_loss_values.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(valset.dataset)
    val_acc_values.append(accuracy)
    
    print('Val Epoch: {}\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(m_epoch,
        loss, correct, len(valset.dataset), accuracy))

def save_graph(epochs:int, attr:list, val_attr:list, title:str)->None:
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(epochs), attr, 'r')
    plt.plot(np.arange(epochs), val_attr, 'b')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(title + ".png")
    plt.close()
    save_path = os.path.join(os.getcwd, title + ".png")
    print(f"Saved {title} graph to: {save_path}.png")

def save_confusion_matrix(cm_input, labels:list, title='Confusion matrix', cmap='Blues'):
    """
    Create a confusion matrix plot
    :param cm_input: np.array of tps, tns, fps and fns
    :param labels: list of class names
    :param title: title of plot
    :param cmap: colormap of confusion matrix
    :return:
    """
    font_size = 14 + len(labels)
    if np.max(cm_input) > 1:
        cm_input = cm_input.astype(int)
    if isinstance(labels[0], str):
        ['\n'.join(wrap(label, 20, break_long_words=False)) for label in labels]
    plt.figure(figsize=(2 * len(labels), 2 * len(labels)))
    plt.imshow(cm_input, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=font_size)

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, rotation_mode="anchor", ha="right", fontsize=font_size)
    plt.yticks(tick_marks, labels, fontsize=font_size)

    thresh = np.max(cm_input) / 2
    for i, j in itertools.product(range(cm_input.shape[0]), range(cm_input.shape[1])):
            plt.text(j, i, "{:d}".format(cm_input[i, j]), horizontalalignment="center",
                     color="white" if cm_input[i, j] > thresh else "black", fontsize=font_size * 2 / 3)
    plt.ylabel('True label', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=font_size)
    plt.tight_layout()
    plt.savefig(title + ".png", bbox_inches='tight')
    plt.close
    save_path = os.path.join(os.getcwd, title + ".png")
    print(f"Saved {title} graph to: {save_path}.png")

def save_classification_report(m_report:dict, title="classification_report")->None:
    save_path = os.path.join(os.getcwd, title + ".json")
    with open(title + ".json", "w+") as f:
        json.dump(m_report, f, indent=2)
    print(f"Classification report saved in: {save_path}")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%d.%m.%y %H:%M:%S',
                            level=logging.DEBUG)
                            
    # kaggle kernels output gauravduttakiit/metal-defect-classification-using-mobilenet-v2 -p /path/to/dest
    traindir = "/content/data/training"
    testdir = "/content/data/validation"
 
    # Hyperparameters
    param_args = {'num_classes': 2,    
                    'Batch_Size': 64,
                    'Learning_Rate': 0.001,       
                    'Epochs': 20 } 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Setup Dataset")
    train_loader, test_loader, labels = setup_dataset(traindir, testdir, param_args['Batch_Size'])

    model = Binary_Classifier(input=3, num_classes=param_args['num_classes']).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=param_args['Learning_Rate'])
    loss = nn.BCELoss() # BinaryCrossEntropy Loss
    train_loss,val_loss, train_acc, val_acc = [], [], [], []
    start = time.time()
    logging.info("Starting training...")
    for epoch in range(param_args['Epochs']):
        train(trainset=train_loader, m_device= device, m_model=model, 
                m_optimizer=optimizer, m_loss=loss, m_epoch=epoch,
                train_loss_values=train_loss,train_acc_values=val_loss)
        validate(valset=test_loader, m_device=device, m_model=model,
                    m_loss=loss, m_epoch=epoch,
                    val_loss_values=val_loss,val_acc_values=val_acc)
    elapsed_time = start - time.time()
    logging.info("End of Training")
    save_graph(param_args['Epochs'], train_loss, val_loss, title='Training-loss')
    save_graph(param_args['Epochs'], train_acc, val_acc, title='Training-accuracy')

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("training duration: {}:{}:{:05.2f}".format(int(hours),int(minutes),seconds))

    # confusion matrix
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for xb_test,yb_test  in test_loader:
            #y_test_pred = model(xb_test.to(device))
            y_pred_tag = torch.round(model(xb_test.to(device)))
            y_pred.append(y_pred_tag.detach().numpy())
            y_true.append(yb_test.numpy())
    y_pred = [a.squeeze().tolist() for a in y_pred]
    y_true = [a.squeeze().tolist() for a in y_true]
    cf_matrix = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cf_matrix, labels)
    report = classification_report(y_true, y_pred, output_dict=True)
    report['training duration'] = "{}:{}:{:05.2f}".format(int(hours),int(minutes),seconds)
    save_classification_report(report)

    
