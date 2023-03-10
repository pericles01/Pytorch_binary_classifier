import torch
import logging
import os
import matplotlib.pyplot as plt
import itertools
import json
from textwrap import wrap
import numpy as np

def save_model(torch_model, path:str = "./binary_model.pth")->None:
    torch.save(torch_model, path)
    logging.info("model saved to {}".format(path))

def save_graph(epochs:int, attr:list, val_attr:list, title:str, type:str="acc")->None:
    label1 = "train_" + type
    label2 = "val_" + type
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(epochs), attr, 'r', label= label1)
    plt.plot(np.arange(epochs), val_attr, 'b', label = label2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(title + ".png")
    plt.close()
    save_path = os.path.join(os.getcwd(), title + ".png")
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
    save_path = os.path.join(os.getcwd(), title + ".png")
    print(f"Saved {title} graph to: {save_path}.png")

def save_classification_report(m_report:dict, title="classification_report")->None:
    save_path = os.path.join(os.getcwd(), title + ".json")
    with open(title + ".json", "w+") as f:
        json.dump(m_report, f, indent=2)
    print(f"Classification report saved in: {save_path}")