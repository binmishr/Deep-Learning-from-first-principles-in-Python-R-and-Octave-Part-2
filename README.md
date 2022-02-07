# Deep-Learning-from-first-principles-in-Python-R-and-Octave-Part-2

The details of the codeset and plots are included in the attached Microsoft Word Document (.docx) file in this repository. 
You need to view the file in "Read Mode" to see the contents properly after downloading the same.

A Brief Introduction
======================

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

*"What does the world outside your head really 'look' like? Not only is there no color, there's also no sound: the compression and expansion of air is picked up by the ears, and turned into electrical signals. The brain then presents these signals to us as mellifluous tones and swishes and clatters and jangles. Reality is also odorless: there's no such thing as smell outside our brains. Molecules floating through the air bind to receptors in our nose and are interpreted as different smells by our brain. The real world is not full of rich sensory events; instead, our brains light up the world with their own sensuality."*
                   The Brain: The Story of You" by David Eagleman
                   
*"The world is Maya,illiusory. The ultimate reality, the Brahman, is all-pervading and all-permeating, which is colourless, odourless, tasteless, nameless and formless"*
                   Bhagavad Gita
                   

     
## 1. Introduction

This post is a follow-up post to my earlier post [Deep Learning from first principles in Python,R and Octave-Part 1]. In the first part, I implemented Logistic Regression, in vectorized Python,R and Octave, with a *wannabe* Neural Network (a Neural Network with no hidden layers). In this second part,I implement a regular, but somewhat *primitive* Neural Network (a Neural  Network with just 1 hidden layer). The 2nd part implements classification of manually created datasets, where the different clusters of the 2 classes are not linearly separable. This post also includes the implemenmtation of a 3 layer Neural Network from first principles. All the Deep Learning gurus suggest that one should start from the basics and be able to tweak the hyperparamaters to get a good understanding the DL networks

## 2.  3 layer Neural Network
A simple representation of a 3 layer Neural Network (NN) with 1 hidden layer is shown below. In this there are 2 input features at the input layer, 3 hidden units at the hidden layer and 1 output layer as it deals with binary classfication. The activation unit at the hidden layer can be a tanh, sigmoid, relu etc. At the output layer the activation is a sigmoid to handle binary classfication

    z_{11} = w_{11}^{1}x_{1} + w_{21}^{1}x_{2} + b_{1}
    z_{12} = w_{12}^{1}x_{1} + w_{22}^{1}x_{2} + b_{1}
    z_{13} = w_{13}^{1}x_{1} + w_{23}^{1}x_{2} + b_{1}

    Also
    a_{11} = tanh(z_{11})
    a_{12} = tanh(z_{12})
    a_{13}} = tanh(z_{13})

    z_{21} = w_{11}^{2}a_{11} + w_{21}^{2}a_{12} + w_{31}^{2}a_{13} + b_{2}
    a_{21} = sigmoid(z21)

    Hence
    \begin{pmatrix}
    z11\\ 
    z12\\ 
    z13
    \end{pmatrix} =\begin{pmatrix}
    w_{11}^{1} & w_{21}^{1} \\ 
    w_{12}^{1} & w_{22}^{1} \\ 
    w_{13}^{1} & w_{23}^{1}
    \end{pmatrix} * \begin{pmatrix}
    x1\\ 
    x2
    \end{pmatrix} + b_{1}\\

    And
    \begin{pmatrix}
    a11\\ 
    a12\\ 
    a13
    \end{pmatrix} = \begin{pmatrix}
    tanh(z11)\\ 
    tanh(z12)\\ 
    tanh(z13)
    \end{pmatrix}

    Similarly 

    z_{21}
     = \begin{pmatrix}
    w_{11}^{2} & w_{21}^{2} & w_{31}^{2}
    \end{pmatrix} *\begin{pmatrix}
    z_{11}\\ 
    z_{12}\\ 
    z_{13}
    \end{pmatrix} +b_{2}

    and
    a_{21} = sigmoid(z_{21})

    These equations can be written as

    Z1 = W1 * X + b1
    A1 = tanh(Z1)
    Z2 = W2 * A1 + b2
    A2 = sigmoid(Z2)



  Some important results (a memory refresher!)
  d/dx(e^{x}) = e^{x}    and d/dx(e^{-x}) = -e^{-x}      -a
  and sinhx = (e^{x} - e^{-x})/2 and coshx = e^{x} + e^{-x})/2 
  Using (a) we can shown that sinh'x = coshx and cosh'x = sinhx        (b)

  Now d/dx(f(x)/g(x)) = (g(x)f'(x) - f(x)g'(x))/g(x)^{2}   -(c)

  Since tanhx = z= sinhx/coshx and using (b) we get
               = (coshx*sinh'x - coshx*sinh'x)/coshx^{2}
  Using the values of the derivative of sinhx and coshx from (b) above we get
    tanhx = z = (coshx^{2} - sinhx^{2})/coshx^{2}
               = 1 - tanhx^{2}
    Since tanhx =z 
    d/dx(tanhx) = 1 - z^{2}            -(d)

    L=-(Y*log(A_{2}) + (1-Y)*log(1-A_{2}))
    L= -(YlogA2 + (1-Y)log(1-A2))
    dL/dA2 = -[Y/A2 + (1-Y)/(1-A2)]

    A2 = sigmoid(Z2)
    dA2/dZ2 = A2(1-A2)


    Z2 = W2A1 +b2

    dZ2/dW2 = A1
    dZ2/db2 = 1

    A1 = tanh(Z1)
    dA1/dZ1 = 1 - A1^{2}

    Z1 = W1X + b1
    dZ1/dW1 = X
    dZ1/db1 = 1

    dL/dA2 = -(Y/A2 + (1-Y)/(1-A2)) 
    dL/dZ2 = dL/dA2 * dA2/dZ2 = -(Y/A2 + (1-Y)/(1-A2)) * A2(1-A2) = A2 - Y
    dL/dW2 = dL/dA2 * dA2/dZ2 * dZ2/dW2 = (A2-Y) *A1 
    dL/db2 = dL/dA2 * dA2/dZ2 * dZ2/db2 = (A2-Y)

    dL/dZ1 = dL/dA2 * dA2/dZ2 * dZ2/dA1 *dA1/dZ1 = (A2-Y) * W2 * (1-A^{2})

    dL/dW1 = dL/dA2 * dA2/dZ2 * dZ2/dA1 *dA1/dZ1 *dZ1/dW1 = 
                       (A2-Y) * W2 * (1-A^{2}) * X

    dL/db1 = dL/dA2 * dA2/dZ2 * dZ2/dA1 *dA1/dZ1 *dZ1/db1 
                      = (A2-Y) * W2 * (1-A^{2}) 

As I mentioned in the earlier post of gradient descent tries  determine  the sets of weights and biases at each layer  which minimize the loss function by traversing the path of steepest descent


## 3.  Manually create a data set that is not linearyly separable
Initially I create a dataset with 2 classes which had around 9 clusters that cannot be separated by linear boundaries
```{python, cache=TRUE}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
import sklearn
import sklearn.datasets
colors=['black','gold']
cmap = matplotlib.colors.ListedColormap(colors)
X, y = make_blobs(n_samples = 400, n_features = 2, centers = 7,
                       cluster_std = 1.3, random_state = 4)
#Create 2 classes
y=y.reshape(400,1)
y = y % 2
plt.figure()
plt.title('Non-linearly separable classes')
plt.scatter(X[:,0], X[:,1], c=y,
           marker= 'o', s=50,cmap=cmap)
plt.savefig('fig1.png', bbox_inches='tight')
```



## 4. Logistic Regression
On the above created dataset, logistic regression is performed and the decision boundary is plotted. It can be seen that logistic regression performs quite poorly
```{python}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
import sklearn
import sklearn.datasets
#from DLfunctions import plot_decision_boundary
execfile("./DLfunctions.py") # Since import does not work in Rmd!!!
colors=['black','gold']
cmap = matplotlib.colors.ListedColormap(colors)
X, y = make_blobs(n_samples = 400, n_features = 2, centers = 7,
                       cluster_std = 1.3, random_state = 4)
#Create 2 classes
y=y.reshape(400,1)
y = y % 2
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X, y);
# Plot the decision boundary for logistic regression
plot_decision_boundary_n(lambda x: clf.predict(x), X.T, y.T,"fig2.png")
```


## 5. 3 layer Neural Network in Python (vectorized)
The vectorized implementation is included below. Note that in the case of Python a learning rate of 0,5 and 3 hidden units performs very well.
```{python}
## Random data set with 9 clusters
#1.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
execfile("./DLfunctions.py") # Since import does not work in Rmd!!!
X1, Y1 = make_blobs(n_samples = 400, n_features = 2, centers = 9,
                       cluster_std = 1.3, random_state = 4)
#Create 2 classes
Y1=Y1.reshape(400,1)
Y1 = Y1 % 2
X2=X1.T
Y2=Y1.T
parameters,costs = computeNN(X2, Y2, numHidden = 4, learningRate=0.5, numIterations = 10000)
plot_decision_boundary(lambda x: predict(parameters, x.T), X2, Y2,str(4),str(0.5),"fig3.png")
```
## 6.  3 layer Neural Network in R (vectorized)
The vectorized implementation of a Neural Network was just a little more interesting as R does not have a similar package like 'numpy'. While numpy handles broadcasting implicitly, in R I had to use the 'sweep' command to broadcast. The implementaion is included below. Note that since the initialization with random weights  is slightly different, R performs best with a learning rate of 0.1 and with 6 hidden units

```{r fig1, cache=TRUE}
source("DLfunctions2_1.R")
z <- as.matrix(read.csv("data.csv",header=FALSE)) # 
x <- z[,1:2]
y <- z[,3]
x1 <- t(x)
y1 <- t(y)
nn <-computeNN(x1, y1, 6, learningRate=0.1,numIterations=10000) # Good
plotDecisionBoundary(z,nn,6,0.1)
```

## 7a.  Performance of the Neural Network for different learning rates (Python)
```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
execfile("./DLfunctions.py") # Since import does not work in Rmd!!!
X1, Y1 = make_blobs(n_samples = 400, n_features = 2, centers = 9,
                       cluster_std = 1.3, random_state = 4)
#Create 2 classes
Y1=Y1.reshape(400,1)
Y1 = Y1 % 2
X2=X1.T
Y2=Y1.T
learningRate=[0.5,1.2,3.0]
df=pd.DataFrame()
for lr in learningRate:
   parameters,costs = computeNN(X2, Y2, numHidden = 4, learningRate=lr, numIterations = 10000)
   print(costs)
   df1=pd.DataFrame(costs)
   df=pd.concat([df,df1],axis=1)
#Set the iterations
iterations=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000]   
#Set index
df1=df.set_index([iterations])
df1.columns=[0.5,1.2,3.0]
fig=df1.plot()
fig=plt.title("Cost vs No of Iterations for different learning rates")
plt.savefig('fig4.png', bbox_inches='tight')
```

## 7b. Performance of the Neural Network for different hidden units (Python)
```{python}
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import pandas as pd
from sklearn.datasets import make_classification, make_blobs
execfile("./DLfunctions.py") # Since import does not work in Rmd!!!
X1, Y1 = make_blobs(n_samples = 400, n_features = 2, centers = 9,
                       cluster_std = 1.3, random_state = 4)
#Create 2 classes
Y1=Y1.reshape(400,1)
Y1 = Y1 % 2
X2=X1.T
Y2=Y1.T
numHidden=[3,5,7]
df=pd.DataFrame()
for numHid in numHidden:
   parameters,costs = computeNN(X2, Y2, numHidden = numHid, learningRate=1.2, numIterations = 10000)
   print(costs)
   df1=pd.DataFrame(costs)
   df=pd.concat([df,df1],axis=1)
#Set the iterations
iterations=[0,1000,2000,3000,4000,5000,6000,7000,8000,9000]   
#Set index
df1=df.set_index([iterations])
df1.columns=[3,5,7]
fig=df1.plot()
fig=plt.title("Cost vs No of Iterations for different no of hidden units")
plt.savefig('fig5.png', bbox_inches='tight')
```

## 8a. Performance of the Neural Network for different learning rates (R)
```{r fig2, cache=TRUE}
source("DLfunctions2_1.R")
z <- as.matrix(read.csv("data.csv",header=FALSE)) # 
x <- z[,1:2]
y <- z[,3]
x1 <- t(x)
y1 <- t(y)
learningRate <-c(0.1,1.2,3.0)
df <- NULL
for(i in seq_along(learningRate)){
   nn <-  computeNN(x1, y1, 6, learningRate=learningRate[i],numIterations=10000) 
   cost <- nn$costs
   df <- cbind(df,cost)
  
}      
df <- data.frame(df) 
iterations=seq(0,10000,by=1000)
df <- cbind(iterations,df)
names(df) <- c("iterations","0.5","1.2","3.0")
library(reshape2)
df1 <- melt(df,id="iterations")    
ggplot(df1) + geom_line(aes(x=iterations,y=value,colour=variable),size=1)  + 
    xlab("Iterations") +
    ylab('Cost') + ggtitle("Cost vs No iterations for  different learning rates")
```

## 8b. Performance of the Neural Network for different hidden units (R)
```{r fig3, cache=TRUE}
source("DLfunctions2_1.R")
# Num hidden
numHidden <-c(4,6,9)
df <- NULL
for(i in seq_along(numHidden)){
    nn <-  computeNN(x1, y1, numHidden[i], learningRate=0.1,numIterations=10000) 
    cost <- nn$costs
    df <- cbind(df,cost)
    
}      
df <- data.frame(df) 
iterations=seq(0,10000,by=1000)
df <- cbind(iterations,df)
names(df) <- c("iterations","4","6","9")
library(reshape2)
df1 <- melt(df,id="iterations")    
ggplot(df1) + geom_line(aes(x=iterations,y=value,colour=variable),size=1)  + 
    xlab("Iterations") +
    ylab('Cost') + ggtitle("Cost vs No iterations for  different number of hidden units")
```

## 9. Turning the heat on the Neural Network
In this 2nd part I create a a central region of positives and and the outside region
as negatives. The points are generated using the equation of a circle (x - a)^{2} + (y -b) ^{2} = R^{2} . How does the 3 layer Neural Network perform on this? Let see below

## 10. Manually creating a circular central region
```{python}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
import sklearn
import sklearn.datasets
colors=['black','gold']
cmap = matplotlib.colors.ListedColormap(colors)
x1=np.random.uniform(0,10,800).reshape(800,1)
x2=np.random.uniform(0,10,800).reshape(800,1)
X=np.append(x1,x2,axis=1)
X.shape
# Create a subset of values where squared is <0,4. Perform ravel() to flatten this vector
a=(np.power(X[:,0]-5,2) + np.power(X[:,1]-5,2) <= 6).ravel()
Y=a.reshape(800,1)
cmap = matplotlib.colors.ListedColormap(colors)
plt.figure()
plt.title('Non-linearly separable classes')
plt.scatter(X[:,0], X[:,1], c=Y,
           marker= 'o', s=15,cmap=cmap)
plt.savefig('fig6.png', bbox_inches='tight')
```

## 11a. Decision boundary with 4 hidden units=4 and learning rate = 2.2 (Python)
With the above hyper parameters the decision boundary is triangular

```{python}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.linear_model
execfile("./DLfunctions.py")
x1=np.random.uniform(0,10,800).reshape(800,1)
x2=np.random.uniform(0,10,800).reshape(800,1)
X=np.append(x1,x2,axis=1)
X.shape
# Create a subset of values where squared is <0,4. Perform ravel() to flatten this vector
a=(np.power(X[:,0]-5,2) + np.power(X[:,1]-5,2) <= 6).ravel()
Y=a.reshape(800,1)
X2=X.T
Y2=Y.T
parameters,costs = computeNN(X2, Y2, numHidden = 4, learningRate=2.2, numIterations = 10000)
plot_decision_boundary(lambda x: predict(parameters, x.T), X2, Y2,str(4),str(2.2),"fig7.png")
```


## 11b. Decision boundary with hidden units=12 and learning rate = 2.2 (Python)
With the above hyper parameters the decision boundary is triangular

```{python}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn.linear_model
execfile("./DLfunctions.py")
x1=np.random.uniform(0,10,800).reshape(800,1)
x2=np.random.uniform(0,10,800).reshape(800,1)
X=np.append(x1,x2,axis=1)
X.shape
# Create a subset of values where squared is <0,4. Perform ravel() to flatten this vector
a=(np.power(X[:,0]-5,2) + np.power(X[:,1]-5,2) <= 6).ravel()
Y=a.reshape(800,1)
X2=X.T
Y2=Y.T
parameters,costs = computeNN(X2, Y2, numHidden = 12, learningRate=2.2, numIterations = 10000)
plot_decision_boundary(lambda x: predict(parameters, x.T), X2, Y2,str(12),str(2.2),"fig8.png")
```


## 12a. Decision boundary with hidden units=9 and learning rate = 0.5 (R)
When the number of hidden units is 6 and the learning rate is 0,1, is also a triangular shape in R
```{r fig4,cache=TRUE}
source("DLfunctions2_1.R")
z <- as.matrix(read.csv("data1.csv",header=FALSE)) # N
x <- z[,1:2]
y <- z[,3]
x1 <- t(x)
y1 <- t(y)
nn <-computeNN(x1, y1, 9, learningRate=0.5,numIterations=10000) # Triangular
plotDecisionBoundary(z,nn,6,0.1)
```

## 12b. Decision boundary with hidden units=8 and learning rate = 0.1 (R)

```{r fig5,cache=TRUE}
source("DLfunctions2_1.R")
z <- as.matrix(read.csv("data1.csv",header=FALSE)) # N
x <- z[,1:2]
y <- z[,3]
x1 <- t(x)
y1 <- t(y)
nn <-computeNN(x1, y1, 8, learningRate=0.1,numIterations=10000) # Hemisphere
plotDecisionBoundary(z,nn,8,0.1)
```
Conclusion: This post implemented a 3 layer Neural Network to create non-linear boundaries while performing classification. Clearly the Neural Network performs very well when the number of hidden units and learning rate are varied.

To be continued.
Watch this space!!
