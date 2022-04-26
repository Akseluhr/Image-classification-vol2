#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:58:48 2022

@author: akseluhr
"""
import numpy as np
import pickle
import numpy.matlib
import matplotlib.pyplot as plt
import sys

###############################################################

# Matlab -> Python functions

###############################################################

# Loades an entire batch
def LoadBatch(filename):
	""" Copied from the dataset website """ 
	with open('Datasets/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes') 
	return dict

# Calculate softmax for class class estimation of each image (vector)
def softmax(x):
	""" Standard definition of the softmax function """
	exp_x = np.exp(x)
	return exp_x / np.sum(exp_x, axis=0)

def ComputeGradsNum(W1, W2, b1, b2, X, Y,lambd, h=0.00001):
    
    grad_W2 = np.zeros(shape=W2.shape)
    grad_b2 = np.zeros(shape=b2.shape)
    grad_W1 = np.zeros(shape=W1.shape)
    grad_b1 = np.zeros(shape=b1.shape)   
    c = ComputeCost(X, Y, W1, W2,b1, b2, lambd)
    
    for i in range(b1.shape[0]):
        b1_try = b1.copy()
        b1_try[i,0] = b1_try[i,0]+h
        c2 = ComputeCost(X, Y, W1, W2,b1_try, b2, lambd)
        grad_b1[i,0] = (c2-c)/h
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = W1.copy()
            W1_try[i,j] = W1_try[i,j]+h
            c2 = ComputeCost(X, Y, W1_try, W2,b1, b2, lambd)
            grad_W1[i,j] = (c2-c)/h
    
    for i in range(b2.shape[0]):
        b2_try = b2.copy()
        b2_try[i,0] = b2_try[i,0]+h
        c2 = ComputeCost(X, Y, W1,W2,b1,  b2_try, lambd)
        grad_b2[i,0] = (c2-c)/h
    
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = W2.copy()
            W2_try[i,j] = W2_try[i,j]+h
            c2 = ComputeCost(X, Y, W1, W2_try,b1, b2, lambd)
            grad_W2[i,j] = (c2-c)/h
    
    return grad_W1,grad_W2,grad_b1,grad_b2

# Allows for efficiently view the images in a directory or 
# in a *Matlab* array or cell array
def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

###############################################################

# My functions

###############################################################

# Read pixel data, labels (classes), one-hot rep. of labels (classes)
# Divide pixel data by 255 for correct format
def ReadData(filename):
    data_batch = LoadBatch(filename)
    pixel_data = data_batch[b'data'].T
    labels = data_batch[b'labels']
    one_hot = np.eye(10)[labels].T
    return pixel_data, one_hot, labels 

# Normalize w.r.t. training data mean and standard deviation
# Normalization of input so that the inputs are at a comparable range
def Normalize(train, validation,test=None):
        train=np.float64(train)
        validation=np.float64(validation)
        
        mean_X =train.mean(axis=1)

        std_X=train.std(axis=1)

        train=train-mean_X[:,None]

        train=train/std_X[:,None]
        validation=validation-mean_X[:,None]
        validation=validation/std_X[:,None]
        
        if(test is not None):
            test=np.float64(test)
            test=test-mean_X[:,None]
            test=test/std_X[:,None]
            return train,validation,test;
        
        return train,validation;

# First init of model params W(eights) and b(ias)
# Init done with 0 mean and 1 / sqrt of d and m
# Random seed for selecting the same rndm numbers for each execution
def GetWeightAndBias(X, Y, m=50):
    
    weights = list()
    bias = list()
    d = X.shape[0]
    k = 10

    std_d = 1 / np.sqrt(d)
    std_m = 1 / np.sqrt(m)
    
    # W1 = m (50) x d (3072)
    # W2 = K (10) x m (50)
    np.random.seed(400)
    weights.append(np.random.normal(loc=0.0, scale=std_d, size=(m, d)))
    weights.append(np.random.normal(loc=0.0, scale=std_m, size=(k, m)))
        
    # b1 = m (50) x 1
    # b2 = K (10) x 1
    np.random.seed(400)
    bias.append(np.random.normal(loc=0.0, scale=std_d, size=(m,1)))#np.zeros(shape=(m, 1))
    bias.append(np.random.normal(loc=0.0, scale=std_m, size=(k,1)))#np.zeros(shape=(b_size[0], 1))
    
    # OLD
    # b = np.random.normal(loc=0.0, scale=0.01, size=(b_size[0], 1))
    # W = np.random.normal(loc=0.0, scale=0.01, size=(10, m))

    return weights, bias

# Evaluation of the network function
# Agan, Softmax returns each probability for each class label
def EvaluateClassifier(X, W1, W2,b1, b2):
   
    # Saving this for later
    #activaiton_list = list() 
    # just 'hard coded' 2 layer NN for now
    # s1 = np.dot(W[0], X) + b[0]
    # Out put of nodes, first layer
    # h1 = np.maximum(0, s1)
    # s2 = np.dot(W[1], h1)+ b[1]
    # Out put of nodes, second layer
    
    s1=W1@X+b1
    #relu    
    h1=s1 * (s1 > 0)
    s2=W2@h1+b2

    P = softmax(s2)
    act_vals = h1
    return P, act_vals

# Total cost of a set of images:
# 1. Regularization term, calculate: lambda * sum(W^2 ij)
# 2. Sum it with l_cross + regularization term -> for each x,y in D
# 3. Multiply everything with 1 / length of D
def ComputeCost(X, Y, W1, W2, b1, b2, lambd):

    # Saving for later
    # Calculate P using softmax
    #P, act_vals = EvaluateClassifier(X, W, b)
    
    # Calculate cross-entropy-loss
    #l_cross = -np.sum(np.multiply(Y, np.log(P)))
    
    # Calculate regularization term
    #sigma = lambda x, k: for i in range(k) np.sum(np.square(x))
    
    #not sure about this one
    # reg_term = lambd * np.sum([np.sum(np.square(w)) for w in W])
    
    # Calculate total cost of the set of imgs
    # J = (1 / len(X[1])) * l_cross + reg_term
    
    P, H = EvaluateClassifier(X, W1, W2, b1, b2)
    lcr =- np.sum(np.multiply(Y, np.log(P)))
    Reg_term = lambd*((W1**2).sum()+(W2**2).sum())
    J = lcr/X.shape[1]+Reg_term
    return J


# Accuracy of the network's predictions
# Percentage of examples for which it gets the correct answer
def ComputeAccuracy(X, y, W1, W2, b1, b2):
    P, act_vals = EvaluateClassifier(X,W1,  W2,b1, b2)
    acc = np.mean(y == np.argmax(P, axis=0))
    
    return acc
    
# Compute gradients of the cost function to see the curve of cost decrease 
# Forward pass is already done since we have already calculated P
def ComputeGradients(act_vals, X, Y, P, W1, W2,lambd):

    n_b1 = X.shape[1]
    K = Y.shape[0]
    d = act_vals.shape[0]

    # Backward pass
    G_batch = -(Y - P)
    
    # Backward pass for W + reg term fix reg later
    grad_W2 = np.dot(G_batch, act_vals.T)/ n_b1 + 2 * lambd * W2
        
    # Backward pass for b
    grad_b2=(np.dot(G_batch,np.ones(shape=(n_b1,1)))/n_b1).reshape(K,1)

    # Backward pass second layer
    G_batch = W2.T @ G_batch

    # g_test = G_batch * act_vals.T
    
    # 1 as in use this node 
    G_batch = G_batch*(act_vals>0) #, where=(act_vals >0))
    
    grad_W1 = np.dot(G_batch, X.T) / n_b1 + 2 * lambd * W1
    grad_b1=(np.dot(G_batch,np.ones(shape=(n_b1,1)))/n_b1).reshape(d,1)


    return grad_W1, grad_W2, grad_b1, grad_b2

# Check if my analytical gradients 
# Using centered difference function
# If the differenc is < 1e-6, the analytical gradients are fine
def CompareGradients(act_vals, X,Y, W1, W2, b1,b2, lambd, threshold):
    
    P, act_vals = EvaluateClassifier(X, W1, W2, b1,b2)

    #Calculate gradients
    grad_W1_a, grad_W2_a, grad_b1_a, grad_b2_a = ComputeGradients(act_vals, X, Y,P, W1,W2, lambd)
    grad_W1_n,grad_W2_n, grad_b1_n, grad_b2_n = ComputeGradsNum(W1, W2, b1,b2, X, Y,lambd, h=0.00001)

    # Calculate differences
    w_rel_error_1 = np.sum(np.abs(grad_W1_a - grad_W1_n)) / np.maximum(0.001, np.sum(np.abs(grad_W1_a) + np.abs(grad_W1_n)))
    w_rel_error_2 = np.sum(np.abs(grad_W2_a - grad_W2_n)) / np.maximum(0.001, np.sum(np.abs(grad_W2_a) + np.abs(grad_W2_n)))

    b_rel_error_1 = np.sum(np.abs(grad_b1_a - grad_b1_n)) / np.maximum(0.001, np.sum(np.abs(grad_b1_a) + np.abs(grad_b1_n)))
    b_rel_error_2 = np.sum(np.abs(grad_b2_a - grad_b2_n)) / np.maximum(0.001, np.sum(np.abs(grad_b2_a) + np.abs(grad_b2_n)))

    # Check differences
    if (w_rel_error_1 and w_rel_error_2) and (b_rel_error_2 and b_rel_error_1) < threshold:
        print("Analytical ok")
    else:
        print("Gradient difference too high")

 
def MiniBatchGD2(X, Y, y, GDparams, W1, W2, b1, b2, X_val=None, Y_val=None, y_val=None, lambd= 0 ):
    n = X.shape[1]
    (eta_min,eta_max,step_size,n_batch,cycles)=GDparams
    metrics = {'updates':[-1], 
               'Loss_scores':[ComputeCost(X, Y, W1, W2, b1, b2, lambd)], 
               'acc_scores':[ComputeAccuracy(X, y, W1, W2, b1, b2)]}
    if X_val is not None:
        metrics['Loss_val_scores'] = [ComputeCost(X_val, Y_val, W1, W2,b1, b2, lambd)]
        metrics['acc_val_scores'] = [ComputeAccuracy(X_val, y_val, W1, W2, b1, b2)]
    batches = dict()

    for j in range(n//n_batch):
            j_start = (j)*n_batch ;
            j_end = (j+1)*n_batch;
            inds = range(j_start,j_end);
            y_batch = [y[index] for index in inds]
            X_batch = X[:, inds];
            Y_batch = Y[:, inds];
            batches[j]=(X_batch,Y_batch,y_batch)
    j = 0
    
    for l in range(cycles):
        for t in range(2*l*step_size, 2*(l+1)*step_size):
            
            if t>= 2*l*step_size and t<(2*l+1)*step_size:
                eta = eta_min+(t-2*l*step_size)/step_size*(eta_max-eta_min)
            elif t>=(2*l+1)*step_size and t<2*(l+1)*step_size:
                eta = eta_max-(t-(2*l+1)*step_size)/step_size*(eta_max-eta_min)

            X_batch, Y_batch, y_batch = batches[j]
            P_batch, H_batch = EvaluateClassifier(X_batch, W1, W2,b1, b2)
            grad_W1, grad_W2, grad_b1, grad_b2 = ComputeGradients(H_batch, X_batch, Y_batch, P_batch, W1, W2,lambd)

          #  print(W1)
           # if(math.isnan(W1[0][0])):
           #     print("NU ÄR DET NAN ")
           #     print("J = ", j)
           #     print(W1)
             #   hej()
                # .1 * 1.5
                # .0001 * 1.5
            W1 -= eta*grad_W1
            b1 -= eta*grad_b1
            W2 -= eta*grad_W2
            b2 -= eta*grad_b2
            j += 1
            if j>(n//n_batch-1):
                # set j = 0 will start new epoch
                j = 0
                metrics['updates'].append(t+1)
                metrics['acc_scores'].append(ComputeAccuracy(X, y, W1, W2, b1, b2))
                metrics['Loss_scores'].append(ComputeCost(X, Y, W1, W2, b1, b2,lambd))

                if X_val is not None:
                    metrics['acc_val_scores'].append(ComputeAccuracy(X_val, y_val, W1, W2,b1, b2))
                    metrics['Loss_val_scores'].append(ComputeCost(X_val, Y_val, W1, W2,b1, b2, lambd))
                message = "In update "+str(t+1)+'/'+str(2*cycles*step_size)+" finishes epoch "+ \
                          str(len(metrics['updates'])-1)+": loss="+str(metrics['Loss_scores'][-1])+ \
                          " and accuracy="+str(metrics['acc_scores'][-1])+" (training set) \r"
                sys.stdout.write(message)
            
        
    
    return W1, b1, W2, b2, metrics


# Read data
X_train, Y_train, y_test = ReadData('data_batch_1')
X_val_train, Y_val_train, y_val_test = ReadData('data_batch_2')
X_test_train, Y_test_train, y_test_test = ReadData('test_batch')

# Normalize all data w.r.t. mean and std of training data
X_train_normalized, X_val_train_normalized, X_test_train_normalized = Normalize(X_train, X_val_train, X_test_train)

# Create model params W and b
W, b = GetWeightAndBias(X_train_normalized, Y_train, m=50)
W1 = W[0]
W2 = W[1]
b1 = b[0]
b2 = b[1]

# Model evaluation (softmax)
P, act_vals = EvaluateClassifier(X_train_normalized, W1, W2, b1, b2)

# Cost function
J = ComputeCost(X_train_normalized, Y_train, W1, W2, b1, b2, lambd = 0.005)
print("Cost: ", J)

# Accuracy
A = ComputeAccuracy(X_train_normalized, y_test, W1, W2, b1, b2)
print("Accuracy: ", A)

# Compute gradients
lmb = 0.05
grad_W1, grad_W2, grad_b1, grad_b2 = ComputeGradients(act_vals, X_train_normalized, 
                                                      Y_train,P,
                                                      W1, W2, lmb)
# Compare numerical gradients with analytical
threshold = 1e-5
CompareGradients(act_vals, X_train_normalized[0:20, [0]],Y_train[:, [0]], 
                 W1[:, 0:20], W2, b1, b2, 0, threshold)

# Plotting metrics
def plot(metrics):
    
    # Cost plot
    fig, ax = plt.subplots()  
    ax.plot(metrics['updates'], metrics['Loss_scores'], 'b', label='Training cost')  
    ax.plot(metrics['updates'], metrics['Loss_val_scores'], 'r', label='Validation cost')  
    ax.set_xlabel('Update step')  
    ax.set_ylabel('Cost')  
    plt.legend()
    plt.show()
    
    # Acc plot 
    fig, ax = plt.subplots()  
    ax.plot(metrics['updates'], metrics['acc_scores'], 'b', label='Training accuracy')  
    ax.plot(metrics['updates'], metrics['acc_val_scores'], 'r', label='Validation accuracy')  
    ax.set_xlabel('Update step')  
    ax.set_ylabel('Accuracy')  
    plt.legend()
    plt.show()
    
def plot_metrics_grid(metrics_grid,lambds):
    
    # Loss plot
    label1='Training: '+str(min(metrics_grid['Loss_scores']))+ ' lambda: '+str(lambds[metrics_grid['Loss_scores'].index(min(metrics_grid['Loss_scores']))])
    label2='Validation: '+str(min(metrics_grid['Loss_val_scores']))+ ' lambda: '+str(lambds[metrics_grid['Loss_val_scores'].index(min(metrics_grid['Loss_val_scores']))])
    plt.plot(lambds,metrics_grid['Loss_scores'], color='b', marker='o',mfc='pink',label=label1 )
    plt.plot(lambds,metrics_grid['Loss_val_scores'], color='r', marker='o',mfc='green',label=label2 )                                                                                 
    plt.ylabel('loss') 
    plt.xlabel('lambda')
    plt.legend(loc="upper right")
    plt.show()
    
    # Accuracy plot
    label1='Training: '+str(max(metrics_grid['acc_scores']))+ ' lambda: '+str(lambds[metrics_grid['acc_scores'].index(max(metrics_grid['acc_scores']))])
    label2='Validation: '+str(max(metrics_grid['acc_val_scores']))+ ' lambda: '+str(lambds[metrics_grid['acc_val_scores'].index(max(metrics_grid['acc_val_scores']))])
    plt.plot(lambds,metrics_grid['acc_scores'], color='b', marker='o',mfc='green',label=label1 )
    plt.plot(lambds,metrics_grid['acc_val_scores'], color='r', marker='o',mfc='magenta',label=label2 )
    plt.ylabel('accuracy') 
    plt.xlabel('lambda') 
    plt.legend(loc="upper right")
    plt.show()
    
###############################################################

# Testing the networks with different params

###############################################################

# Test 1 hyperparams
lambd = 0
eta_min=1e-5
eta_max=1e-1
step_size=500
n_batch=10
cycles=2
GDparams=(eta_min,eta_max,step_size,n_batch,cycles)

# Run the test
W1_upd, b1_upd, W2_upd, b2_upd, metrics = MiniBatchGD2(X_train_normalized[:, 0:100], Y_train[:, 0:100], y_test[0:100], GDparams, 
                                                       W1, W2, b1, b2,  
                                                       X_val_train_normalized[:, 0:100], 
                                                       Y_val_train[:, 0:100], y_val_test[0:100], lambd)

# Plotting first test
plot(metrics)

###############################################################

# Test 2

###############################################################

# Test 2 hyperparams
lambd = 0.01
eta_min=1e-5
eta_max=1e-1
step_size=500
n_batch=100
cycles=1 
GDparams=(eta_min,eta_max,step_size,n_batch,cycles)

# Re-init the weights 
W, b = GetWeightAndBias(X_train_normalized, Y_train, m=50)
W1 = W[0]
W2 = W[1]
b1 = b[0]
b2 = b[1]

# Run the test
W1_upd, b1_upd, W2_upd, b2_upd, metrics = MiniBatchGD2(X_train_normalized, Y_train, y_test, GDparams, 
                                                       W1, W2, b1, b2,  
                                                       X_val_train_normalized, 
                                                       Y_val_train, y_val_test, lambd)

# Plotting second test
plot(metrics)

###############################################################

# Test 3

###############################################################

# Test 3 hyperparams
lambd = 0.01
eta_min=1e-5
eta_max=1e-1
step_size=800
n_batch=100
cycles=3 
GDparams=(eta_min,eta_max,step_size,n_batch,cycles)

# Re-init the weights
W, b = GetWeightAndBias(X_train_normalized, Y_train, m=50)
W1 = W[0]
W2 = W[1]
b1 = b[0]
b2 = b[1]

# Run the test
W1_upd, b1_upd, W2_upd, b2_upd, metrics = MiniBatchGD2(X_train_normalized, Y_train, y_test, GDparams, 
                                                       W1, W2, b1, b2,  
                                                       X_val_train_normalized, 
                                                       Y_val_train, y_val_test, lambd)
# Plot third test
plot(metrics)

###############################################################

# Test 4 with grid searches and more data

###############################################################

def gridsearchCV(lambds, hidden_dimension, X, Y, y, GDparams, X_val=None,
                Y_val=None, y_val=None):
    metrics_grid = {'Loss_scores':[], 'acc_scores':[]}
    
    # Include validation data if there is some
    if X_val is not None:
        metrics_grid['Loss_val_scores'] = []
        metrics_grid['acc_val_scores'] = []
    
    for lambd in lambds:
        
        # Re-init weights for each lambda value
        W,b=GetWeightAndBias(X, Y, m=50)
        W1 = W[0]
        W2 = W[1]
        b1 = b[0]
        b2 = b[1]
        
        # Build model with current lambda value 
        W1_4, b1_4, W2_4, b2_4, metrics = MiniBatchGD2(X, Y, y, GDparams, W1, W2, b1, b2,X_val,Y_val,y_val, lambd)
        
        # Append to the metrics
        metrics_grid['Loss_scores'].append(metrics['Loss_scores'][-1])
        metrics_grid['acc_scores'].append(metrics['acc_scores'][-1])
        
        # Include validation data if there is some
        if X_val is not None:
            metrics_grid['Loss_val_scores'].append(metrics['Loss_val_scores'][-1])
            metrics_grid['acc_val_scores'].append(metrics['acc_val_scores'][-1])
    return metrics_grid

# Read all data
X1,Y1,y1=ReadData('data_batch_1')
X2,Y2,y2=ReadData('data_batch_2')
X3,Y3,y3=ReadData('data_batch_3')
X4,Y4,y4=ReadData('data_batch_4')
X5,Y5,y5=ReadData('data_batch_5')

# Stack it up
X_new = np.hstack((X1, X2, X3, X4, X5))
Y_new = np.hstack((Y1, Y2, Y3, Y4,Y5))
y_new = y1+y2+y3+y4+y5

# Choose 5k random instances for new validation data
# (More data = less overfitting)
np.random.seed(0)
rand_indexes = np.random.choice(range(X_new.shape[1]), 5000, replace=False)
X_val_new = X_new[:,rand_indexes]
Y_val_new = Y_new[:,rand_indexes]
y_val_new = [y_new[i] for i in rand_indexes]
y_new = [y_new[i] for i in range(X_new.shape[1]) if i not in rand_indexes]
X_new = np.delete(X_new, rand_indexes, 1)
Y_new = np.delete(Y_new, rand_indexes, 1)

# Normalize the new data
X_new, X_val_new, X_test_new = Normalize(X_new, X_val_new, X_test_train)

# Random lambda values
np.random.seed(0)
l_max, l_min = -1, -5
l = l_min+(l_max-l_min)*np.random.rand(8)
print(l)
lambds = list(10**l)
lambds.sort()
print(lambds)

# New params
eta_min=1e-5
eta_max=1e-1
step_size=800
n_batch=100
cycles=2 
GDparams=(eta_min,eta_max,step_size,n_batch,cycles)

# Run test with random lambdas
metrics_grid=gridsearchCV(lambds, 50, X_new, Y_new, y_new, GDparams, X_val_new,
                Y_val_new, y_val_new)

# Plot of the random search
plot_metrics_grid(metrics_grid, lambds)

# Defining fine search w.r.t. results of the random search
lambdas_list=[0.00025,0.0005,0.001,0.0015,0.002,0.0025,0.003,0.005]
print(lambdas_list)

# Run test with best lambdas 
metrics_grid=gridsearchCV(lambdas_list, 50, X_new, Y_new, y_new, GDparams, X_val_new,
                Y_val_new, y_val_new)

# Plot of the fine search
plot_metrics_grid(metrics_grid,lambdas_list)

###############################################################

# Final test 
# On test data using the best lambda

###############################################################

# Final test hyperparams
lambd = 0.002
eta_min=1e-5
eta_max=1e-1
step_size=900
n_batch=100
cycles=3 

# Re-init weights
W, b = GetWeightAndBias(X_train_normalized, Y_train, m=50)
W1 = W[0]
W2 = W[1]
b1 = b[0]
b2 = b[1]

# Run the test
W1_upd, b1_upd, W2_upd, b2_upd, metrics = MiniBatchGD2(X_new, Y_new, y_new, GDparams, 
                                                       W1, W2, b1, b2,  
                                                       X_val_new,Y_val_new, y_val_new, lambd)

# Plot it 
plot(metrics)

# Predict + compute acc for test dataset
A_final = ComputeAccuracy(X_test_new, y_test_test, W1_upd, W2_upd, b1_upd, b2_upd)

# Plot the final accuracy
print("Final Accuracy on test set: ", A_final)
