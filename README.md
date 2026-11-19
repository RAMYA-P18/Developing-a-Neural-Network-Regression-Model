# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features.

In many real-world applications—such as house price prediction, temperature forecasting, sales estimation, or demand prediction—the relationship between input variables and the output is non-linear and complex. Traditional statistical models often fail to capture these patterns effectively. Deep Learning models, particularly Artificial Neural Networks (ANNs), are capable of learning such complex relationships through multiple hidden layers and non-linear activation functions.

In this experiment, a dataset containing multiple independent variables (features) and a dependent variable (target) is provided. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

The experiment aims to understand how network architecture, learning rate, number of epochs, and activation functions affect the accuracy of regression predictions and to demonstrate the effectiveness of deep learning in solving regression problems.

## Neural Network Model
Include the neural network model diagram.
<img width="1028" height="986" alt="image" src="https://github.com/user-attachments/assets/11f520b9-2423-400b-b0df-da921cc64bfd" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Ramya P

### Register Number: 212223240137

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('/content/EXP1.csv')
X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values
print(X)
print(y)
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```
```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}
  def forward(self, x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x
```
```
# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)
# Name:DHAN VARSHINI J P
# Register Number:212224230055
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()





        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
```
```
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

```
```
loss_df = pd.DataFrame(ai_brain.history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

```
```
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```
### Dataset Information
<img width="230" height="598" alt="image" src="https://github.com/user-attachments/assets/542a0904-efb8-46ef-8f36-71993df0ee9b" />


### OUTPUT
<img width="416" height="317" alt="image" src="https://github.com/user-attachments/assets/b23108c5-fac6-4ab7-a452-a8e3cc02e750" />

### Training Loss Vs Iteration Plot
<img width="497" height="365" alt="image" src="https://github.com/user-attachments/assets/a81c34fc-41b2-43b4-81f9-5e4998cd57c2" />


### New Sample Data Prediction
<img width="332" height="47" alt="image" src="https://github.com/user-attachments/assets/fee01e87-408c-4bb3-ac6c-4bfffac0efb7" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
