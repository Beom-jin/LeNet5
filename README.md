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
**Developed LeNet ( DLeNet5) : 99.15 %**

