## Autoencoders

## Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

## Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

## Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype = 'int')

## Getting the total number of users and movies
total_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
total_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))



## function for creating matrix for both test and train set

def updating(data):
    new_data = []
    for i in range(1, total_users + 1):
        rated_movies = data[:, 1][data[:, 0] == i]
        rated_ratings = data[:, 2][data[:, 0] == i]
        
        ratings = np.zeros(total_movies)
        
        ratings[rated_movies - 1] = rated_ratings
        
        new_data.append(list(ratings))
        
    return new_data
    
training_set = updating(training_set)
test_set = updating(test_set)


## Converting it ot torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


## Creating the architecture of the neural networks
class SAE(nn.Module):
    ## for inheritance we use , 
    def __init__(self, ):
        super(SAE, self).__init__()
        ## here we are just specifying the number of neurons
        ## and creating the structure based on stacked autoencoders
        ## encoding
        self.fc1 = nn.Linear(total_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        ##decoding
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, total_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        ## applying activation to the hidden nodes(right ones in linear lines)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        ## here we dont have any hidden neuron layer to apply activation to
        x = self.fc4(x)
        return x
        
sae = SAE()
criterian = nn.MSELoss()
## sae.parameters will return weights and biases 
## weight decay will shrink the weights so that overfitting or underfitting data problems can be resolved
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


## Training the SAE
nb_epochs = 200
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(total_users):
        ## a way to give a dimension to batch
        ## as pytorch requires a batch all the time even if its be trained on 1 inp
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            ## this is running forward we dont need to put .forward we can et it without as...one of the benefits of pytorch
            ## output is getting the predicted result
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterian(output, target)
            ## all that stuff at the end means to not have 0 in denomenator
            mean_corrector = total_movies/float(torch.sum(target.data > 0) + 1e-10)
            ## this tells the model how to update the weights
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            ## this specifies with how much intensity the weights be updated or the amount of weights
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
    
## Testing the SAE
test_loss = 0
s = 0.
for id_user in range(total_users):
    ## training set contains the ratings of the users for particular movies which can be considered the rated movies in the past
    ## test set are the ratings of movies which were done in the future
    ## here inp will contain training_set ratings because the model needs to predict the ratings for that particular user who have nt watched the particular movie
    ## the test_set will then be able to compare with that predicted result to compare the past predicted and future actual ratings of that movie to see how well the model worked
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterian(output, target)
        mean_corrector = total_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item()*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

            

