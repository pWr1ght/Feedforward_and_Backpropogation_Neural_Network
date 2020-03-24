This is a feedforward/backpropogation neural network utilizes many different classes without the use of pre-made machine learning frameworks. (i.e. tensorflow, pytorch) 
This Neural network uses these libraries:
- Matplotlib 
- Mlxtend
- Numpy

If the code is cloned or used, a suggestion would be to use the pycharm IDE. It keeps all the graphs in a contained queue as there are many graphs produced.

What can this thing do:

- Interactive simple terminal menu with plenty of options.
- Provides three simple datasets which are Iris, Sin, XOR.
- Creates a scatterplot for training and testing Sin dataset.
- Creates a confusion matrix for testing Iris dataset.
- Allows user to test/train data
- Gives the abilities to set the amount of nodes, layers, and learning rate per layer.
- Allows the user to traverse through the layers once the neural network is built.

Here is a diagram of the class structures in the neural network indicated below:
Note: Arrows indicate the inheritance.

<img width="694" alt="Screen Shot 2020-03-21 at 3 23 38 PM" src="https://user-images.githubusercontent.com/44282168/77237813-92276100-6b88-11ea-87df-e75cdfd74272.png">


Here is an example of the training 50% of the sin dataset put at 10001 epochs, 1 hidden layer (learning rate .05) with three nodes.


0 epochs

<img width="499" alt="Screen Shot 2020-03-24 at 9 24 05 AM" src="https://user-images.githubusercontent.com/44282168/77451363-cb8ee500-6db1-11ea-9298-5fe6f17b6404.png">



10001 epochs


<img width="497" alt="Screen Shot 2020-03-24 at 9 27 02 AM" src="https://user-images.githubusercontent.com/44282168/77451385-d21d5c80-6db1-11ea-941e-c16dc0139d3f.png">



The training RSME is 0.02530.

Testing the Network with the rest of the dataset. (other 50%)


<img width="498" alt="Screen Shot 2020-03-24 at 9 27 33 AM" src="https://user-images.githubusercontent.com/44282168/77451399-d8abd400-6db1-11ea-874c-5b5724078a0d.png">



The testing RSME is 0.02944


Now testing with the Iris dataset with the same amount of nodes, layers, amount trained, and learning rate.

Confusion matrix of the rest of the data being tested.

<img width="499" alt="Screen Shot 2020-03-24 at 9 49 25 AM" src="https://user-images.githubusercontent.com/44282168/77453957-2a099280-6db5-11ea-856b-8c9e286931c4.png">

The testing RSME is 0.2819

This confusion matrix can be quite misleading for verifying predictions becuase there were some instances that had low prediction scores for each classification. So I decided to make it so the highest classification score within the instance was going to chosen as the predicted label. 

