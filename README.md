# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   NETWORK MODEL:
<img width="1183" height="467" alt="image" src="https://github.com/user-attachments/assets/a11674f4-3aea-4a9a-ab0d-dd01508b5228" />


## DESIGN STEPS
### STEP 1: 
Import PyTorch libraries and define the CNNClassifier class with convolutional, pooling, and fully connected layers.
### STEP 2: 
Initialize the model, define the loss function (CrossEntropyLoss), and set up the optimizer (Adam).
### STEP 3: 
Pass input images through convolution + ReLU + max pooling layers inside the forward() method.
### STEP 4: 
Flatten the feature maps and pass them through fully connected layers to get final class outputs.
### STEP 5: 
In training, perform forward pass, compute loss, apply backpropagation (loss.backward()), and update weights using optimizer.step().
### STEP 6: 
Repeat for multiple epochs and print the final epoch loss along with name and register number.




## PROGRAM

### Name:FRANKLIN RAJ G

### Register Number:212223230058

```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    print('Name: franklin raj g')
    print('Register Number:212223230058')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch
<img width="1427" height="87" alt="image" src="https://github.com/user-attachments/assets/ca4e7db5-ae8a-4f68-bc56-beae0d34d226" />


## Confusion Matrix
<img width="946" height="647" alt="image" src="https://github.com/user-attachments/assets/2f786cdf-2663-4ae7-886e-393f3691e942" />


## Classification Report
<img width="618" height="324" alt="image" src="https://github.com/user-attachments/assets/7563defa-c400-4fd6-8739-804308e30570" />


### New Sample Data Prediction
<img width="657" height="467" alt="image" src="https://github.com/user-attachments/assets/5e631d8d-5a71-48f8-97b0-19fbb6e7e823" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify
the response for new images is executed and verified successfully
