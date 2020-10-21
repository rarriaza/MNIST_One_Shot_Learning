# MNIST_One_Shot_Learning
Implementation of one shot learning with siamese architecture to classify MNIST when 1 sample per class is available.

To run the full code:

python run_all.py

If the MNIST dataset was not downloaded, it will and the trainig data will the pre-process so just 1 sample per class is available. 
The rest of the samples from the training set are not used. 

Settings.py contains all the parameters with their default values. If it is necesary to change one of the parameters, it is possible by adding to the 
command --parameter new_value:

python run_all.py --parameter new_value

Two architectures are trained. One of them is the Embedding network which aims to find a vector-representation of each class, the other is the classifier
which receives the vector and it associate it to one class. More details about these architectures can be found in Networks.py.


