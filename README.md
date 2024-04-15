# homework
### Report 1 - 3
Using the following function, the number of parameters in the model was measured.

~~~
total_params = sum(p.numel() for p in model.parameters())
~~~

**LeNet5 ( Params : 64,122) : 98.97 %**  
**Custom Mlp ( Params : 63,980) : 97.38%**
#### Train Accruacy / Loss 
![Train_Accuracy](https://github.com/Beom-jin/LeNet5/assets/87561766/5534b1ff-5c53-435f-8c0c-2b6ef2112730)
![Train_Loss](https://github.com/Beom-jin/LeNet5/assets/87561766/0e38dc0b-577f-4dd5-9cb8-1e775366dd55)
#### Test Accuracy / Loss 
![Test Accuracy](https://github.com/Beom-jin/LeNet5/assets/87561766/4f926d20-e178-4438-a5a3-35d3b2e44ba2)
![Test Loss](https://github.com/Beom-jin/LeNet5/assets/87561766/550ac678-639c-41ea-86d6-0f43f791ee0e)



### Report 4
Utilize weight decay and batch normalization method
[1] Weight Decay = 0.001 (optimizer)
~~~
optimizer_3 = torch.optim.SGD(model_3.parameters(), lr=0.01,momentum=0.9,weight_decay=0.001)
~~~
[2] Batch Normalization 
~~~
class DLeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(DLeNet5, self).__init__()
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5,stride=1,padding=2)
        self.bn_1 = nn.BatchNorm2d(6) 
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) 
        self.bn_2 = nn.BatchNorm2d(16) 
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.conv3 = nn.Conv2d(6, 16, kernel_size=5) 
        self.bn_3 = nn.BatchNorm2d(16) 
        self.MaxPool_3 = nn.MaxPool2d(kernel_size=(2,2),stride=2)

        self.conv4 = nn.Conv2d(16, 120, kernel_size=5) 
        self.bn_4 = nn.BatchNorm2d(120) 
        self.FC_1 = nn.Linear(120,84)
        self.FC_2 = nn.Linear(84,10)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.MaxPool_1(self.activation(self.bn_1(self.conv1(x))))

        x_1 = self.MaxPool_2(self.activation(self.bn_2(self.conv2(x))))
        x_2 = self.MaxPool_3(self.activation(self.bn_3(self.conv3(x))))
        x = x_1+x_2
        x = self.activation(self.bn_4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.activation(self.FC_1(x))
        x = self.softmax(self.FC_2(x))
        return x
~~~
**Developed LeNet ( DLeNet5) : 99.15 %**

