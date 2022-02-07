# Deep-Learning-from-first-principles-in-Python-R-and-Octave-Part-2

The details of the codeset and plots are included in the attached Microsoft Word Document (.docx) file in this repository. 
You need to view the file in "Read Mode" to see the contents properly after downloading the same.

A Brief Introduction
======================

A Brief Introduction
======================

**Random Initialization**
This technique just initializes the weights to small random values based on Gaussian or uniform distribution


```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
import sklearn
import sklearn.datasets
exec(open("DLfunctions61.py").read())
#Load the data
train_X, train_Y, test_X, test_Y = load_dataset()
# Set the layers dimensions
layersDimensions = [2,7,1]

# Train a deep learning network with random initialization
parameters = L_Layer_DeepModel(train_X, train_Y, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="sigmoid",learningRate = 0.6, num_iterations = 9000, initType="default", print_cost = True,figure="fig1.png")

# Clear the plot
plt.clf()
plt.close()

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), train_X, train_Y,str(0.6),figure1="fig2.png")
```


**He Initialization**
He initialization multiply the random weights by \sqrt{\frac{2}{dimension\ of\ previous\ layer}}

```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
import sklearn
import sklearn.datasets
exec(open("DLfunctions61.py").read())

#Load the data
train_X, train_Y, test_X, test_Y = load_dataset()
# Set the layers dimensions
layersDimensions = [2,7,1]

# Train a deep learning network with He  initialization
parameters = L_Layer_DeepModel(train_X, train_Y, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="sigmoid", learningRate =0.6,    num_iterations = 10000,initType="He",print_cost = True,                           figure="fig3.png")

plt.clf()
plt.close()
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), train_X, train_Y,str(0.6),figure1="fig4.png")
```


**Xavier Initialization**
Xavier initialization multiply the random weights by \sqrt{\frac{1}{dimension\ of\ previous\ layer}

```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
import sklearn
import sklearn.datasets
exec(open("DLfunctions61.py").read())

#Load the data
train_X, train_Y, test_X, test_Y = load_dataset()
# Set the layers dimensions
layersDimensions = [2,7,1]
 
parameters = L_Layer_DeepModel(train_X, train_Y, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="sigmoid",
                            learningRate = 0.6,num_iterations = 10000, initType="Xavier",print_cost = True,
                            figure="fig5.png")

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), train_X, train_Y,str(0.6),figure1="fig6.png")
```


## Default initialization - R
```{r fig1, cache=TRUE}
source("DLfunctions61.R")
z <- as.matrix(read.csv("circles.csv",header=FALSE)) 

x <- z[,1:2]
y <- z[,3]
X <- t(x)
Y <- t(y)
layersDimensions = c(2,11,1)
retvals = L_Layer_DeepModel(X, Y, layersDimensions,
                            hiddenActivationFunc='relu',
                            outputActivationFunc="sigmoid",
                            learningRate = 0.5,
                            numIterations = 9000, 
                            initType="default",
                            print_cost = True)
#Plot the cost vs iterations
iterations <- seq(0,9000,1000)
costs=retvals$costs
df=data.frame(iterations,costs)
ggplot(df,aes(x=iterations,y=costs)) + geom_point() + geom_line(color="blue") +
 ggtitle("Costs vs iterations") + xlab("No of iterations") + ylab("Cost")

# Plot the decision boundary
plotDecisionBoundary(z,retvals,hiddenActivationFunc="relu",lr=0.5)

```



## He initialization
```{r fig 2,cache=TRUE}
source("DLfunctions61.R")
z <- as.matrix(read.csv("circles.csv",header=FALSE)) 

x <- z[,1:2]
y <- z[,3]
X <- t(x)
Y <- t(y)
layersDimensions = c(2,11,1)
retvals = L_Layer_DeepModel(X, Y, layersDimensions,
                            hiddenActivationFunc='relu',
                            outputActivationFunc="sigmoid",
                            learningRate = 0.5,
                            numIterations = 9000, 
                            initType="He",
                            print_cost = True)

#Plot the cost vs iterations
iterations <- seq(0,9000,1000)
costs=retvals$costs
df=data.frame(iterations,costs)
ggplot(df,aes(x=iterations,y=costs)) + geom_point() + geom_line(color="blue") +
    ggtitle("Costs vs iterations") + xlab("No of iterations") + ylab("Cost")

# Plot the decision boundary
plotDecisionBoundary(z,retvals,hiddenActivationFunc="relu",0.5,lr=0.5)

```

## Xavier initialization
```{r fig3, cache=TRUE}
## Xav initialization 
layersDimensions = c(2,11,1)
retvals = L_Layer_DeepModel(X, Y, layersDimensions,
                            hiddenActivationFunc='relu',
                            outputActivationFunc="sigmoid",
                            learningRate = 0.5,
                            numIterations = 9000, 
                            initType="Xav",
                            print_cost = True)

#Plot the cost vs iterations
iterations <- seq(0,9000,1000)
costs=retvals$costs
df=data.frame(iterations,costs)
ggplot(df,aes(x=iterations,y=costs)) + geom_point() + geom_line(color="blue") +
    ggtitle("Costs vs iterations") + xlab("No of iterations") + ylab("Cost")

# Plot the decision boundary
plotDecisionBoundary(z,retvals,hiddenActivationFunc="relu",0.5)
```


##Regularization - Python

```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
import sklearn
import sklearn.datasets
exec(open("DLfunctions61.py").read())

#Load the data
train_X, train_Y, test_X, test_Y = load_dataset()
# Set the layers dimensions
layersDimensions = [2,7,1]

# Train a deep learning network
parameters = L_Layer_DeepModel(train_X, train_Y, layersDimensions, hiddenActivationFunc='relu',  
                               outputActivationFunc="sigmoid",learningRate = 0.6, lambd=0.1, num_iterations = 9000, 
                               initType="default", print_cost = True,figure="fig7.png")

# Clear the plot
plt.clf()
plt.close()

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), train_X, train_Y,str(0.6),figure1="fig8.png")


plt.clf()
plt.close()
plot_decision_boundary(lambda x: predict(parameters, x.T,keep_prob=0.9), train_X, train_Y,str(2.2),"fig8.png",)
```


## Spiral data Regularization 2 - Python
```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
import sklearn
import sklearn.datasets
exec(open("DLfunctions61.py").read())
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j


# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.clf()
plt.close()  
layersDimensions = [2,100,3]
y1=y.reshape(-1,1).T
parameters = L_Layer_DeepModel(X.T, y1, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="softmax",
                           learningRate = 0.6,lambd=0.2, num_iterations = 5000, print_cost = True,figure="fig9.png")

plt.clf()
plt.close()  
W1=parameters['W1']
b1=parameters['b1']
W2=parameters['W2']
b2=parameters['b2']
plot_decision_boundary1(X, y1,W1,b1,W2,b2,figure2="fig10.png")
```


```{r fig4, cache=TRUE}
source("DLfunctions61.R")
df=read.csv("circles.csv",header=FALSE)

z <- as.matrix(read.csv("circles.csv",header=FALSE)) 

x <- z[,1:2]
y <- z[,3]
X <- t(x)
Y <- t(y)
layersDimensions = c(2,11,1)
retvals = L_Layer_DeepModel(X, Y, layersDimensions,
                            hiddenActivationFunc='relu',
                            outputActivationFunc="sigmoid",
                            learningRate = 0.5,
                            lambd=0.1,
                            numIterations = 9000, 
                            initType="default",
                            print_cost = True)

#Plot the cost vs iterations
iterations <- seq(0,9000,1000)
costs=retvals$costs
df=data.frame(iterations,costs)
ggplot(df,aes(x=iterations,y=costs)) + geom_point() + geom_line(color="blue") +
    ggtitle("Costs vs iterations") + xlab("No of iterations") + ylab("Cost")


# Plot the decision boundary
plotDecisionBoundary(z,retvals,hiddenActivationFunc="relu",0.5)
```


## Regularization 2
```{r fig5, cache=TRUE}
# Read the spiral dataset
source("DLfunctions61.R")
Z <- as.matrix(read.csv("spiral.csv",header=FALSE)) 

# Setup the data
X <- Z[,1:2]
y <- Z[,3]
X <- t(X)
Y <- t(y)

layersDimensions = c(2, 15, 6, 3) #lr 2.6 20K, 0.15
retvals = L_Layer_DeepModel(X, Y, layersDimensions,
                            hiddenActivationFunc='relu',
                            outputActivationFunc="softmax",
                            learningRate = 5.1,
                            lambd=0.1,
                            numIterations = 9000, 
                            print_cost = True)



parameters<-retvals$parameters
plotDecisionBoundary1(Z,parameters)
```

## Dropout 1 - Python
```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
import sklearn
import sklearn.datasets
exec(open("DLfunctions61.py").read())
#Load the data
train_X, train_Y, test_X, test_Y = load_dataset()
# Set the layers dimensions
layersDimensions = [2,7,1]

# Train a deep learning network
parameters = L_Layer_DeepModel(train_X, train_Y, layersDimensions, hiddenActivationFunc='relu',  
                               outputActivationFunc="sigmoid",learningRate = 0.6, keep_prob=0.7, num_iterations = 9000, 
                               initType="default", print_cost = True,figure="fig11.png")

# Clear the plot
plt.clf()
plt.close()

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T,keep_prob=0.7), train_X, train_Y,str(0.6),figure1="fig12.png")

```


### Dropout 2 - Python
```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
import sklearn
import sklearn.datasets
exec(open("DLfunctions61.py").read())
# Create an input data set - Taken from CS231n Convolutional Neural networks,
# http://cs231n.github.io/neural-networks-case-study/
               

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j


# Plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.clf()
plt.close()  
layersDimensions = [2,100,3]
y1=y.reshape(-1,1).T
parameters = L_Layer_DeepModel(X.T, y1, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="softmax",
                           learningRate = 0.6,keep_prob=0.5, num_iterations = 5000, print_cost = True,figure="fig13.png")

plt.clf()
plt.close()  
W1=parameters['W1']
b1=parameters['b1']
W2=parameters['W2']
b2=parameters['b2']
plot_decision_boundary1(X, y1,W1,b1,W2,b2,figure2="fig14.png")
```


## Dropout - R
```{r fig7, cache=TRUE}
source("DLfunctions61.R")
df=read.csv("circles.csv",header=FALSE)

z <- as.matrix(read.csv("circles.csv",header=FALSE)) 

x <- z[,1:2]
y <- z[,3]
X <- t(x)
Y <- t(y)
layersDimensions = c(2,11,1)
retvals = L_Layer_DeepModel(X, Y, layersDimensions,
                            hiddenActivationFunc='relu',
                            outputActivationFunc="sigmoid",
                            learningRate = 0.5,
                            keep_prob=0.8,
                            numIterations = 9000, 
                            initType="default",
                            print_cost = True)

# Plot the decision boundary
plotDecisionBoundary(z,retvals,keep_prob=0.6, hiddenActivationFunc="relu",0.5)
```



```{r fig8, cache=TRUE}
# Read the spiral dataset
source("DLfunctions61.R")
Z <- as.matrix(read.csv("spiral.csv",header=FALSE)) 

# Setup the data
X <- Z[,1:2]
y <- Z[,3]
X <- t(X)
Y <- t(y)

source("DLfunctions61.R")
layersDimensions = c(2, 15, 6, 3)
retvals = L_Layer_DeepModel(X, Y, layersDimensions,
                            hiddenActivationFunc='relu',
                            outputActivationFunc="softmax",
                            learningRate = 5.1,
                            keep_prob=0.8,
                            numIterations = 9000, 
                            print_cost = True)

parameters<-retvals$parameters
plotDecisionBoundary1(Z,parameters)

```


