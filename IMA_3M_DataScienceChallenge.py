"""
IMA Math-to-Industry Bootcamp, 2018
3M Data Science Challenge
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#### Change point detection

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



def plot_der(i,beg,fin,scale,df,der,k=1):
    """
    Just to investigate, we write a function that takes the range of data, the number of the sensor (3 to 18)
    and plots its derivative approximations against the concentration levels. Note that k is a dummy variable holding the name
    of the figure.
        
    i the number of the sensor's column in df=df_smooth
    beg is the beginning of the data range
    fin is the end of the data range
    scale is the integer we scale the derivative by
    der is df_der
    k is the figure number, set to 1 if not specified.
    """
    x=df.T.values[0] #timestamp
    y=df.T.values[1] #CO concentration
    z=50*df.T.values[2] #etholene concentration
    deriv=scale*der.T.values[i] #derivatives scaled by some integer for easier viewing
    x=x[beg:fin]
    y=y[beg:fin]
    deriv=deriv[beg:fin]
    z=z[beg:fin]
    
    plt.figure(k,figsize=(24,16))    #Plot with the given range
    plt.plot(x,y,color="blue",label='CO')    
    plt.plot(x,z,color="orange",label='Ethylene')
    plt.plot(x,deriv,color="green",label='Derivative approximation')
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Concentration(ppm)',fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    #plt.suptitle('Sensor %d'%(i), fontsize=25)

    

#### Model training and prediction visualization.

def plot_to_original(CO_pred,eth_pred,timesteps,df):
    """
    This function takes the predictions for each time chunk and plots in its
    the original scale.
    
    CO_pred and eth_pred are arrays of length m storing the predictions for each time-chunk
    timesteps is an array of length m+1 with the change-points
    
    df 'reads' df_smooth
    """
    
    subseries=df[(df.timestamp>timesteps[0]) & (df.timestamp<timesteps[len(timesteps)-1])]
    
    t=timesteps[0:len(timesteps)-1] #We don't need the last point for plotting 
    ytrue=subseries['CO_conc']
    ztrue=subseries['ethylene_conc']
    time=subseries['timestamp']


    x=[k+26 for k in t] #Fast forward by 26


    plt.figure(1, figsize=(30,20))
    plt.step(x,CO_pred,color='red',label='CO_prediction', linewidth=4)
    plt.plot(time,ytrue,color='blue',label='true CO values',linewidth=4)
    plt.legend(fontsize=50)

    plt.figure(2, figsize=(30,20))
    plt.step(x,eth_pred,color='purple',label='ethylene prediction', linewidth=4)
    plt.plot(time,ztrue,color='orange',label='true ethylene values',linewidth=4)
    plt.legend(fontsize=50)
    plt.show()

def svr_plot(X,y,z, ker='rbf', supress=False):
    '''
    Standard vector regression with radial basis kernel unless otherwise specified.
    X is the feature matrix (m x n)
    y is the CO response (m dimensional vector)
    z is the ehtylene response (m dimensional vector)
    If supress is true, the plots are supressed
    '''
    
    from sklearn.svm import SVR
    from sklearn import preprocessing

    #Standardize data
    min_max_scaler = preprocessing.MinMaxScaler()
    X_st = min_max_scaler.fit_transform(X)
    y_st=min_max_scaler.fit_transform(np.asarray(y).reshape(-1,1))
    z_st=min_max_scaler.fit_transform(np.asarray(z).reshape(-1,1))


    X_train=X_st[0:238] #60% training data
    X_test=X_st[238:,:] #test data
    
    y_train=y_st[:238]
    y_test=y_st[238:]
    z_train=z_st[:238]
    z_test=z_st[238:]


    clf = SVR(kernel=ker,C=1.0, epsilon=200) #rbf kernel, other methods include linear and poly
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    
    clf = SVR(kernel=ker,C=1.0, epsilon=200)
    clf.fit(X_train,z_train)
    z_pred=clf.predict(X_test)
    
    plt.figure(1,figsize=(10,8))
    
    plt.plot(range(238,395),y_test,color='blue',label='true CO values')
    plt.plot(range(238,395),y_pred,color='red',label='predicted CO values')
    plt.ylabel('Concentration (ppm)')
    plt.xlabel('Timeseries id')
    plt.legend(fontsize=16)
    
    plt.figure(2,figsize=(12,8))
    plt.plot(range(238,395),z_test,color='orange',label='true ethylene values')
    plt.plot(range(238,395),z_pred,color='purple',label='predicted ethylene values')
    plt.ylabel('Concentration (ppm)')
    plt.xlabel('Timeseries id')
    plt.legend(fontsize=16)
    
    plt.show()
    

def lasso_plot(X, y,z,supress=False):
    """
    1-D Lasso
    
    # X is the feature matrix
    # y is the CO vector
    # z is the ethylene response
    # if supress is True, no plot are made
    """
    
    from sklearn import linear_model

    reg = linear_model.Lasso(alpha=1200) #Create model object
    
    n=int(len(y)*0.3)
    
    if isinstance(X,pd.core.frame.DataFrame):
        X=np.asarray(X)   
    X_test=X[:n,:]
    y_test=y[:n] 
    z_test=z[:n]
    X_train=X[n:,:]
    y_train=y[n:]
    z_train=z[n:]
    
    
    reg.fit(X_train, y_train) #Fit the model with CO_response
    y_pred=reg.predict(X_test)

    #y=[x+reg.intercept_ for x in y] #Add intercept
    #Note to self: this is a nice little trick

    if supress==False:
        #Plot against true values
        plt.figure(1,figsize=(30,20))
        plt.plot(y_pred,color='red')
        plt.plot(y_test,color='blue')
        plt.legend(['CO_pred', 'CO_true'],fontsize=36)

        #Now repeat with ethylene
        reg = linear_model.Lasso(alpha=1200) #Create model object
        reg.fit(X_train, z_train) #Fit the model with CO_response
        z_pred=reg.predict(X_test)

        plt.figure(2,figsize=(30,20))
        plt.plot(z_pred,color='purple')
        plt.plot(z_test,color='orange')
        plt.legend(['eth_pred', 'eth_true'],fontsize=36)

        plt.show()
    
    return [y_test,y_pred,z_test,z_pred]


def multi_lasso_plot(X,CO_response,ethylene_response,supress=False):
    """
    Multi Task Lasso
    
    # X is the (m x n) feature vector. This can be an array or a pandas data frame.
    # CO_response is the m dimensional vector with the true CO values
    # ethylene_response is the m dimensional vector with the true ethylene values    
    
    #If suppress is true, the plots are supressed.
    """
    
    from sklearn import linear_model
    
    # Create an array-type matrix with CO and ethylene values.
    # This is just the format MultiTaskLasso takes    
    y=[]
    for i in range(0,395):
        y.append([CO_response[i],ethylene_response[i]])    
    
    model2 = linear_model.MultiTaskLasso(alpha=1200) #Create MultiTaskLasso object.
    
    n=int(len(CO_response)*0.7) #70 % for training
    
    if isinstance(X,pd.core.frame.DataFrame):
        X=np.asarray(X)   
    X_train=X[:n,:]
    y_train=y[:n] 
    X_test=X[n:,:]
    y_test=y[n:]
    
    model2.fit(X_train, y_train)
    
    ypred=model2.predict(X_test)

    #ypred=np.matmul(np.asarray(X),model2.coef_.T) #Multiply matrices out

    #Collect predicted values
    co_pred=[]
    eth_pred=[]
    for i in range(0,len(ypred)):
        co_pred.append(ypred[i][0])
        eth_pred.append(ypred[i][1])

    # Add intercepts
    #co_pred=[x+model2.intercept_[0] for x in co_pred]
    #eth_pred=[x+model2.intercept_[1] for x in eth_pred]
    
    if supress==False :
        plt.figure(1,figsize=(30,20))
        plt.plot(CO_response[n:],color='blue')
        plt.plot(co_pred,color='red')
        plt.legend(['CO_pred', 'CO_true'],fontsize=36)

        plt.figure(2,figsize=(30,20))
        plt.plot(ethylene_response[n:],color='orange')
        plt.plot(eth_pred,color='purple')
        plt.legend(['eth_pred', 'eth_true'],fontsize=36)

        plt.show()
    
    return [co_pred,eth_pred]
    
def elastic_net_plot(X,y,z,a):
    # X is the feature vector (m x n) array, n is the number of features
    # y and z are array like objects of length m
    # a is the alpha parameter for the elastic net

    from sklearn.linear_model import ElasticNet
    from sklearn import preprocessing

    #Standardize data
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    y=min_max_scaler.fit_transform(np.asarray(y).reshape(-1,1))
    z=min_max_scaler.fit_transform(np.asarray(y).reshape(-1,1))

    
    
    X_train=X[:int(0.6*len(y)),:] #60% for training
    y_train=y[:int(0.6*len(y)),:]
    z_train=z[:int(0.6*len(y)),:]

    X_test=X[int(0.6*len(y)):,:]
    y_test=y[int(0.6*len(y)):,:]
    z_test=z[int(0.6*len(y)):,:]

    regr = ElasticNet(random_state=0,alpha=a,normalize=False)
    regr.fit(X_train, y_train)
    regr2=ElasticNet(random_state=0,alpha=a,normalize=False)
    regr2.fit(X_train,z_train)


    y_pred=regr.predict(X_test)
    z_pred=regr2.predict(X_test)
    
    #Co_plot
    plt.figure(1,figsize=(20,10))
    plt.plot(range(int(0.6*len(y)),len(y)),y_test)
    plt.plot(range(int(0.6*len(y)),len(y)),y_pred,color='red')
    
    #Ethylene plot
    plt.figure(2,figsize=(20,10))
    plt.plot(range(int(0.6*len(y)),len(y)),z_test,color='orange')
    plt.plot(range(int(0.6*len(y)),len(y)),z_pred,color='purple')
    plt.show()

    
def random_tree_plot(X,y,z,maxd,k=1,supress=False):
    # We write a function that fits and plots a random tree model
    # X is the feature matrix
    # y is the CO_vector
    # z is the ethylene vector
    # maxd is the maximum depth
    # k is the number of the figure
    # If supress is True, not plots are made

    from sklearn.tree import DecisionTreeRegressor

    n=int(0.6*len(y))
    X_train=np.asarray(X)[:n,:]
    y_train=y[:n]
    z_train=z[:n]
    X_test=np.asarray(X)[n:,:]
    y_test=y[n:]
    z_test=z[n:]

    # Fit regression model
    regr = DecisionTreeRegressor(splitter='random', criterion='friedman_mse',max_depth=maxd)
    regr2= DecisionTreeRegressor(splitter='random', criterion='friedman_mse',max_depth=maxd)
    regr.fit(X_train, y_train)
    regr2.fit(X_train, z_train)

    # Predict
    y_pred = regr.predict(X_test)
    z_pred = regr2.predict(X_test)
    
    if supress==False:
        # Plot the results
        plt.figure(k,figsize=(12,8))
        plt.plot(y_pred, color="yellowgreen", label="max_depth=%d"%(maxd), linewidth=2)
        plt.plot(y_test,color='cornflowerblue',linewidth=2,label='true CO values')
        #plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Decision Tree Regression")
        plt.legend(fontsize=16)
    
        plt.figure(k+1,figsize=(12,8))
        plt.plot(z_pred, color="purple", label="max_depth=%d"%(maxd), linewidth=2)
        plt.plot(z_test,color='orange',linewidth=2,label='true ethylene values')
        #plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Decision Tree Regression")
        plt.legend(fontsize=16)
    
    return [y_pred,z_pred]
    
    
    plt.show()
    
# This is a little function that splits data into training and testing data in the given ratio.
def test_train(X,y,ratio):
    n=int(ratio*len(y))
    if isinstance(X,pd.core.frame.DataFrame): #convert to array if needed
        X=np.asarray(X)        
    X_train=X[:n,:]
    X_test=X[n:,:]
    y_train=y[:n]
    y_test=y[n:]
    return [X_train,X_test,y_train,y_test]

def random_forest_plot(X,y,z,maxd,k=1,supress=False):
    # X is the m x n feature vector
    # y and z are n dimensional arrays with the response variables
    # k is just the number of the figure incase we want to plot multiple ones.
    # maxd is the maximum depth
    # If supress is true, no plots are made.

    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train,y_test=test_train(X,y,0.7)
    X_train, X_test, z_train,z_test=test_train(X,z,0.7)

    # CO prediction
    regr = RandomForestRegressor(max_depth=maxd,max_features='auto',random_state=0)
    regr.fit(X_train, y_train)
    y_pred=regr.predict(X_test)
    
    # Ethylene prediction
    regr = RandomForestRegressor(max_depth=maxd,max_features='auto',random_state=0)
    regr.fit(X_train, z_train)
    z_pred=regr.predict(X_test)

    if supress==False:
        plt.figure(1,figsize=(12,8))
        plt.subplot(2,1,1)
        plt.plot(y_test,label='true value',color='blue')
        plt.plot(y_pred,color='red',label='max_depth %d'%(maxd))
        plt.legend()
    
        plt.subplot(2,1,2)
        plt.plot(z_test,label='true value',color='purple')
        plt.plot(z_pred,color='orange',label='max_depth %d'%(maxd))
        plt.legend()
    
    
        plt.show()
    
    return [y_pred,z_pred]

def kernel_ridge_plot(X,y,ker='linear'):
    #kernel ridge regression, kernel linear unless otherwise specified
    
    from sklearn.kernel_ridge import KernelRidge
    X_train, X_test, y_train,y_test=test_train(X,y,0.7)

    clf = KernelRidge(alpha=1.0,kernel=ker)
    # There are many options for kernel here: rbf, linear, poly, sigmoid...

    clf.fit(X_train, y_train)

    y_pred=clf.predict(X_test)
    plt.figure(figsize=(24,16))
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.show()

