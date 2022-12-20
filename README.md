# Pytorch_binary_classifier

<p> Pytorch CNN project for binary classification task. </p>
<p> The CNN is built from scratch with 3 convelution blocks and 2 fully connected layers. <p>
Each convelution block consist of a sequence of:
<li> 1 convelution layer </li>
<li> 1 batch normalization layer </li>
<li> relu activation </li>
<li> 1 dropout layer </li>


See [./Model.py](https://github.com/pericles01/Pytorch_binary_classifier/blob/main/Model.py)

# Installation
- Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) on your machine if you don't have it
- Clone the git project with 
  ```bash 
   git clone https://github.com/pericles01/Pytorch_binary_classifier.git
   cd Pytorch_binary_classifier
  ```
- Run the shell code below to install the project dependencies
  ```bash
   conda env create -f ./environment.yml
  ```
- Activate the virtual environment
    ```bash
    conda activate TorchEnv
    ```
    Can be deactivted after the use with ``conda deactivate``
# Dataset
Class names and files' names are user defined, please keep the class numbering as shown
The positive class name must start with the digit: 00_ and the negative class name with: 01_
```
dataset/
        00_class_name/
                img_1.png
                img_2.jpg
                ...
        01_class_name/
                img_3.png
                img_4.jpg
                ...
```
Set the input and test datasets' path in [./main.py](https://github.com/pericles01/Pytorch_binary_classifier/blob/main/main.py)
# Run
- for windows
    ```bash
    python main.py
    ```
- for Linux
    ```bash
    python3 main.py
    ```
# Test
Some files are created and save in the current working directory:
- Loss, accuracy and confusion matrix graphes (png files)
- A classification report (json file) 
