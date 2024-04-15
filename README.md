# homework
### Report 1 - 3
Using the following function, the number of parameters in the model was measured.

~~~
total_params = sum(p.numel() for p in model.parameters())
~~~

LeNet5 ( Params : 64,122) : 98.97 % 
Custom Mlp ( Params : 63,980) : 97.38%



### Report 4
Utilize weight decay and batch normalization method
Developed LeNet ( DLeNet5) : 99.15 %

