# NLP A5
 AIT NLP Assignment 5

- [Student Information](#student-information)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Extra features](#extra-features)


## Student Information
Name - Minn Banya  
ID - st124145

## Installation and Setup
Webapp at localhost:8000

## Usage
cd into app folder and run 'python3 app.py'  

## Model evaluation and comparison
| Model       | Training Loss | Average Cosine Similarity | Accuracy |
|------------------|---------------|--------------|--------------|
|S-BERT (pretrained)|        1.169303         |       0.7661      | 34.1%         |
| S-BERT(scratch) |         2.707229          |      0.9985       | 32.85%        |

Although at first glance, our model's performance looks good, this is due to the fact that it is predicting `<PAD>` for all masked tokens. As such, when we fine tune for S-BERT, the two sentences are always showing as similar, even when they are not. This makes our model's performance unapplicable for usage.

Where as, using the pretrained `BERT base uncased` gives a very standard but satisfactory result when applying the training method mentioned in the S-BERT paper.

The pre-trained model performed better with an accuracy of 34.1% and loss of 1.17.

## Impact of hyperparameters
The hyperparameters chosen for training our BERT model was:  
Number of Encoder of Encoder Layer - 6  
Number of heads in Multi-Head Attention - 8  
Embedding Size/ Hidden Dim - 768  
Number of epochs - 10  
Training data - 1000000 sentences  
Vocab size - 73276  

The hyperparameters chosen for tuning S-BERT on our BERT model was:
Training data - 10000 rows  
Number of epochs - 1  

I believe the poor performance of our model is due to the limitation of the training data size, as well as having only 6 layers of encoders. The vocab size 
