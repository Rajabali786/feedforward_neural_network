import torch
import torch.nn as nn  
import torch.nn.functional as F 
import pandas as pd 
import matplotlib.pyplot as plt



# create a model class that inherists nn.module 
# input layer (4 feature of flowers)
# 2 hidden layers (nummb of neurons)
# out put (3 classes if ireas flowers)
class Model(nn.Module):
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x
  
 # pick a manual seed for randomization 
torch.manual_seed(32)
  # create a insrance of model 
model= Model()

# check the data and feature 
url=('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
my_df= pd.read_csv(url)

'''print(my_df.tail())'''

# change last column from string to integers 
my_df['variety']= my_df['variety'].replace('setosa',0.0)
my_df['variety']=my_df['variety'].replace('varsicolor',1.0)
my_df['variety']=my_df['variety'].replace('virginica',2.0)
print(my_df)



#  train test split set x , y
x= my_df.drop('variety',axis=1)
y= my_df['variety']

# convert these to numpy arrays
x= x.values
y= y.values

'''print(x)       # check the all data 
print(y)'''


from sklearn.model_selection import train_test_split


# Assume x and y are your dataframes/numpy arrays
x = my_df.drop('variety', axis=1).apply(pd.to_numeric).values
y = my_df['variety'].replace({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}).values

# Train test split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)

# Check data types and shapes before converting to tensors
print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Convert x features to float tensors
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

# Convert y labels to tensor long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# Set the criterion of the model to measure the error
criterion = nn.CrossEntropyLoss()

# lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# chect the feature how does it works..........
print( 'this is model parameter' ,  model.parameters)



#.................................................................
#..............TRAIN OUR MODEL.....................
# EPOCHS?  (one run through all the training data in our network)
epochs=100
losses= []
for i in range(epochs):
 # go forward and get a prediction 
 y_pred= model.forward(x_train)    # get predicted results 

 # measure the loss/error, gonna be hight at first 
 loss= criterion(y_pred,y_train)   # predicted values vs the y_train

 # keep track of our losses
 losses.append(loss.detach().numpy())

 # print every 5 epoch 
 if i % 10 ==0:
  print(f'Epoch: {i} and  loss:{loss}  ') 


# do some back propagation: take the error rate of forward propagation and feed its back
# thru the network to fine tune the weights 
 optimizer.zero_grad()         # this line space is very important  
 loss.backward()
 optimizer.step() 


# Graph it out!
plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')

'''plt.show()''' # show the diagram '''   


# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
  y_eval = model.forward(x_test) # X_test are features from our test set, y_eval will be predictions
  loss = criterion(y_eval, y_test) # Find the loss or error

  print(loss)

correct = 0
with torch.no_grad():
  for i, data in enumerate(x_test):
    y_val = model.forward(data)

    if y_test[i] == 0:
      x = "Setosa"
    elif y_test[i] == 1:
      x = 'Versicolor'
    else:
      x = 'Virginica'


    # Will tell us what type of flower class our network thinks it is
    print(f'{i+1}.)  {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

    # Correct or not
    if y_val.argmax().item() == y_test[i]:
      correct +=1

print(f'We got {correct} correct!')


# tell the data they will predict you what kind of ires is this ..........
new_iris = torch.tensor([15.1,	-2.8,	94.7,	-100.2])

# Add .unsqueeze(0) to make it a batch of one sample
with torch.no_grad():
    output = model(new_iris.unsqueeze(0))  # Pass the new iris data

# Find the predicted class (the index of the highest score in output)
predicted_class = output.argmax().item()

# Map the predicted class to the flower name
flower_types = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Print the result
print(f"This iris flower is most likely: {flower_types[predicted_class]}")



#  save the NN model 
torch.save(model.state_dict(), 'This_is_my_first_NN_Network')

# load the save model 
new_model= Model()
new_model.load_state_dict(torch.load('This_is_my_first_NN_Network'))

# make sure its loaded correctly  
new_model.eval()



#  this is my first neural network ......................

