# LangNeuralNetwork
Custom neural network that learns to predicts language of a given input String The network will train on the languages selected in the array 'testing[]'. Language data has been automatically romanized. LangN provides means of training a multi-layer network, as well as optimizers (momentum and RMSProp). Lang1 will train a single layer network.

## Command Documentation:

**train** *int*: trains the neural network for given number of iterations

**test** *string*: feeds given string into neural network and outputs values

**trial** *string*/*int*: runs test data for selected language (selected via whole name, three letter abreviation or corresponding number)

**alltrial**: runs test data for all languages in testing[] (all languages the network is training on)

**togdetail**: toggle printing advanced data (full predictions)

**togcount**: toggle printing count

**togacc**: toggle printing accuracy

**togstep**: (new step): changes step size of neural network

