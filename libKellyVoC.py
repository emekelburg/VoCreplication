# 2024-02-13  EM
#  fixed issue in volstdbwd to use sample standard deviation vs. population standard deviation

# 2024-02-10  EM
#  updated getPlist function to fix a number of P's and calculate based on log scale with escalating distances
#  updated fit metrics function to incorporate maximum loss

# 2024-02-09  EM
#  added output of both scaled and unscaled to simulation and benchmark functions

# 2024-02-08  EM
#  updated density of P-list and fixed issue with sequencing of P-values

# 2024-02-07  EM
#  Added scaling factor output to volstdbwd
#  updated Ridge and adjusted Ridge function to de-standardize predictions
#  added cumulative-returns to fit metrics output
#  simplified P-list generation to fewer permutations
#  updated OLS calculation in fit stats

# 2024-02-06 Erik Mekelburg
#  Completed the first version of Python port of Kelly Virtue of Complexity Functionality

import numpy as np
from time import time

def volstdbwd(K_, window=None):
    """
    Standardizes an array by volatility using an expanding or rolling window.  
    If no window is provided, using expanding window with minimum 36 observations.

    Parameters:
    K_ : numpy.ndarray
        The input matrix to be standardized.
    window : int, optional
        The window size for rolling standardization. If None, uses an expanding window.

    Returns:
    Kout : numpy.ndarray
        The standardized matrix.
    Kscale:  the scaling factors for each period - used to unscale predictions
    """
    T = K_.shape[0]
    Kout = np.full_like(K_, np.nan)
    Kscale = np.full_like(K_, np.nan)

    if window is None: # expanding window of size 36+
        for t in range(T):
            if t < 36:
                Ktmp = K_[:36, :]
                Kfactor = np.nanstd(Ktmp, axis=0, ddof=1)
                Kout[:36, :] = Ktmp / Kfactor #np.nanstd(Ktmp, axis=0)
                Kscale[:36, :] = Kfactor

            else:
                Ktmp = K_[:t + 1, :]
                Kfactor = np.nanstd(Ktmp, axis=0, ddof=1)
                Kout[t, :] = K_[t, :] / Kfactor #np.nanstd(Ktmp, axis=0)
                Kscale[t, :] = Kfactor
    else: # rolling window of side T
        for t in range(T):
            if t < window:
                Ktmp = K_[:window, :]
                Kfactor = np.nanstd(Ktmp, axis=0, ddof=1)
                Kout[:window, :] = Ktmp / Kfactor #np.nanstd(Ktmp, axis=0)
                Kscale[:window, :] = Kfactor
            else:
                Ktmp = K_[t - window + 1:t + 1, :]
                Kfactor = np.nanstd(Ktmp, axis=0, ddof=1)
                Kout[t, :] = K_[t, :] / Kfactor #np.nanstd(Ktmp, axis=0)
                Kscale[t, :] = Kfactor

    return Kout, Kscale


def calculate_betas_using_standard_ridge(Y, X, lambda_list):
    """
    Perform ridge regression using Singular Value Decomposition.
    
    Parameters:
    Y : numpy array of shape (T, 1)
    X : numpy array of shape (T, P)
    lambda_ : numpy array of shape (L, 1) or (L,)
    
    Returns:
    B : numpy array of shape (P, L)
    """
    if np.isnan(X).any() or np.isnan(Y).any():
        raise ValueError("missing data")
    
    #print(f"X:{X}")
    
    T, P = X.shape
    
    #assert T >= P

    L = len(lambda_list)
    # Performs Singular Value Decomposition on the predictor matrix X. 
    #   Vt is the transpose of V from MATLAB's SVD
    U, D, Vt = np.linalg.svd(X, full_matrices=False)
    #D = s
    
    B = np.empty((P, L))

    for index_, lambda_ in enumerate(lambda_list):
        # Construct the diagonal matrix used in ridge regression coefficient calculation, 
        #   incorporating the regularization parameter.
        diag_s = np.diag(D / (D**2 + lambda_))
        # Calculates the coefficients for the current value of lambda_ using 
        #  the formula derived from the ridge regression solution in matrix form
        B[:, index_] = (Vt.T @ diag_s @ U.T @ Y).flatten()  # we transpose Vt because of the Matlab/Python difference

    return B


def calculate_betas_using_Kelly_method(Y, X, lambda_list):
    """
    Calculate regression coefficients using a custom method when there are more parameters than observations.
    
    Parameters:
    Y : numpy.ndarray
        The dependent variable vector, shape (T, 1).  T is the number of observations.
    X : numpy.ndarray
        The independent variable matrix, shape (T, P).  T is the number of observations, P the number of parameters.
    lambda_list : numpy.ndarray
        An array of regularization parameters, shape (L,).
    
    Returns:
    B : numpy.ndarray
        Regression coefficients for each lambda, shape (P, L).
    """
    if np.isnan(X).any() or np.isnan(Y).any():
        raise ValueError('missing data')
    
    L_ = len(lambda_list)
    
    T_, P_ = X.shape  # T is the number of observations, P is the number of parameters
    
    # this function should only be run for cases where P > T, otherwise use regular Ridge regression
    assert P_ > T_
    
   # if P_ > T_: # more parameters than observations
        # computes the covariance matrix of the observations, 
        #   scaled by the number of observations. The result is a T×T matrix. 
        #  This scaling factor (T_) normalizes the covariance matrix, making it essentially the 
        #  average covariance across all observations
        #   WHY:  For high-dimensional data (P>T), where the number of predictors exceeds the number of observations, 
        #  it's often more computationally efficient or numerically stable to work with a smaller T×T matrix rather 
        #  than a larger P×P matrix.
    a_matrix = X @ X.T / T_  # Shape: T x T
    
   # else: # less than or equal parameters to observations
  #      # Computes the covariance matrix of the predictors, scaled by the number of observations. 
  #      #  The result in this case is a P×P matrix. Similar to the first case, this scaling normalizes the covariance matrix.
  #      #  For lower-dimensional data (P≤T), working with a P×P matrix is straightforward and aligns with many standard 
  #      #  statistical methods, such as principal component analysis (PCA) or ridge regression.
  #      a_matrix = X.T @ X / T_  # Shape: P x P
    
    # Decomposes this covariance matrix into its eigenvalues (s) and eigenvectors (U_a and V_a). 
    #  The eigenvalues represent the variance explained by each principal component, 
    #  while the eigenvectors represent the directions of maximum variance in the data space. 
    #  This decomposition is crucial for understanding the structure of the data and for further 
    #  calculations in the get_beta function, particularly for regularization and dimensionality reduction purposes.
    U_a, s, V_a_transpose = np.linalg.svd(a_matrix, full_matrices=True)
    # Convert the singular values into a diagonal matrix for D_a
    #D_a = np.diag(s)
    # Note: NumPy returns V transposed, so we need to transpose it for multiplication purposes
    V_a = V_a_transpose.T
    
    # make eigenvalues (amount of variance explained) into a scaled matrix with same dimensions as a_matrix
    #scale_eigval = np.diag((D_a * T_) ** (-1/2)) 
    # explicitly computes the inverse square root only for non-zero eigenvalues, 
    #  effectively bypassing the division by zero issue for zero eigenvalues by setting their inverse square roots to zero.
    inv_sqrt_eigvals = np.zeros_like(s)
    nonzero_indices = s > 0  # Find indices of non-zero eigenvalues
    inv_sqrt_eigvals[nonzero_indices] = 1.0 / np.sqrt(s[nonzero_indices] * T_)
    scale_eigval = np.diag(inv_sqrt_eigvals)
    
    # W is constructed by multiplying X' (transpose of X, making it PxT), U_a (from the SVD, dimension depends on a_matrix), 
    #   and a scaling matrix derived from D_a (scale_eigval, which applies an inverse square root scaling to the eigenvalues). 
    #  This operation aims to transform the predictors into a space where the regularization will be applied.
    W = X.T @ U_a @ scale_eigval 
    
    # vector of dimension Px1 representing the correlation (or "signal") between each predictor and the response variable, 
    #  scaled by the number of observations
    signal_times_return = X.T @ Y / T_   # P×1 since X' is P×T and Y is T×1
    
    # adjust the signal vector into the transformed predictor space
    signal_times_return_times_v = W.T @ signal_times_return   # PxT if P>T and PxP otherwise

    B = np.empty((P_, L_))
    
    for index_, lambda_ in enumerate(lambda_list):   
        # Adjust the transformed predictors (W) by the regularization term. 
        #   This operation scales each transformed predictor component based on 
        #   its eigenvalue and the regularization parameter
        regularized_predictors = W @ np.diag(1/(s+lambda_))
        coefficients = regularized_predictors @ signal_times_return_times_v
        B[:, index_] = coefficients.flatten() 

    return B


def get_lambda_list():
    return 10.0 ** np.arange(-3, 4)  # list of lambdas for ridge regression

#def get_Plist(trnwin = 12, maxP=12000):
    #return np.concatenate([
    #    np.array([2]),
    #    np.arange(5, trnwin, max(1, trnwin // 10)),
    #    np.arange(trnwin - 4, trnwin + 5, 2),
    #    np.arange(trnwin + 5, trnwin * 30, max(1, trnwin // 2)),
    #    np.arange(trnwin * 31, maxP, trnwin * 10),
    #    np.array([maxP])
    #])
#    return np.concatenate([
#        np.array([2,5]),
#        #np.arange(5, trnwin, max(1, trnwin // 2)),
#        np.arange(trnwin - 4, trnwin + 5, 2),
#        np.arange(trnwin + 10, trnwin * 99, max(1, trnwin * 5 )),
#        np.arange(trnwin * 100, maxP, trnwin * 110),
#        np.array([maxP])
#    ])

def get_Plist(max_P=12000, num_points = 30):
    # Starting point (P_start) and ending point (P_end)
    P_start = 2
    P_end = max_P

    # Generate the numbers on a log scale
    numbers = np.logspace(np.log10(P_start), np.log10(P_end), num_points)

    # Convert to integers with rounding and ensure uniqueness and minimum step size of 2
    unique_numbers = [int(numbers[0])]
    for num in numbers[1:]:
        next_int = max(unique_numbers[-1] + 2, int(round(num)))
        if next_int not in unique_numbers:
            unique_numbers.append(next_int)

    # Remove duplicates while preserving order (in case adjustments cause reversions)
    numbers_int = list(dict.fromkeys(unique_numbers))

    # Ensure the final number does not exceed P_end
    numbers_int = [num for num in numbers_int if num <= P_end]

    return numbers_int


def Kelly_rff_ridge_sim(X, Y, iSim, maxP = 12000, gamma = 2, trnwin = 12, stdize = True): #, rescale_predictions = True):
    """
    Computes OOS forecasts with one random seed for many permutations of complexity and different 
     Ridge shrinkage lambdas.
    
    Parameters:
    X : numpy.array
        M<ulti-dimensional numpy array of predictor variabes, Shape T x d with T observations of d variables.
    Y : numpy.array
        Outcome variable.   Shape T x 1.
    iSim : int
        Random seed for this simulation.
    maxP : maximum number of parameters generated by Random Fourier Features
    gamma : float
        Gamma in Random Fourier Features.
    trnwin : int
        Training window size.
    stdize : bool
        Standardization flag. True to apply variance standardization of X and Y.
    rescale_predictions : bool
        scale predictions back to original scale
    """

    start_time = time()
    nSim = 1  # Total number of simulations run in this function

    # Choices of complexity - build up the list of options from 2 to 12000
    #maxP = 12000  # provided as parameter to the function
    #Plist = np.concatenate([
    #    np.array([2]),
    #    np.arange(5, trnwin, max(1, trnwin // 10)),
    #    np.arange(trnwin - 4, trnwin + 5, 2),
    #    np.arange(trnwin + 5, trnwin * 30, max(1, trnwin // 2)),
    #    np.arange(trnwin * 31, maxP, trnwin * 10),
    #    np.array([maxP])
    #])
    #Plist = get_Plist(trnwin=trnwin, maxP = 12000)
    Plist = get_Plist( max_P = maxP)
    print(f"Plist: {Plist}")

    trainfrq = 1 # re-train ridge model at every step
    lamlist = get_lambda_list() # 10.0 ** np.arange(-3, 4)  # list of lambdas for ridge regression
    demean = False # don't demean the variables in the out of sample run

    nL = len(lamlist)  # different lambdas for ridge
    nP = len(Plist)  # number of RFF parameters

    # load the dataset
    #GYdata = loadmat('GYdata.mat')
    #X, Y, dates = GYdata['X'], GYdata['Y'], GYdata['dates']

    # Add lag return (Y variable) as a predictor
    X = np.hstack([X, np.roll(Y, 1)])

    # Vol-standardize
    if stdize:
        # standardize X and Y by past 12 months' volatility, 
        #   computed as the square root of the average of the squared lags up to 12 months back.
        X, _ = volstdbwd(X)  # standardize X # uses expanding as default   
        Y, Y_scale_std = volstdbwd(Y, window=12)  # standardize Y  
        
        # Drop first 3 years due to vol scaling of X
        Y = Y[36:]
        Y_scale_std = Y_scale_std[36:] # discard the scaling factors for the first three years as well
        X = X[36:, :]
 
        #dates = dates[36:]

    T, d = X.shape # T is number of observations and d is number of variables
    X = X.T # transpose X
    Y = Y.reshape(-1, 1) # ensure Y is a two-dimensional array with one column (this is superfluous)

    # Empty Output Space based on number of Observations (T) RFF parameters (nP), number of lambdas (nL), number of simulations (nSim)
    Yprd = np.full((T, nP, nL, nSim), np.nan)
    Bnrm = np.full((T, nP, nL, nSim), np.nan)

    # Recursive Estimation
    np.random.seed(iSim)
    # Fix random features for maxP - maximum number of parameters, then slice data
    # W is the matrix of random Gaussian weights 
    W = np.random.randn(maxP, d) # generate maxP times random weights for all given underlying parameters

    print(f"W: {len(W[0])}")

    # loop through every number of RFF parameters
    for p, P in enumerate(Plist):
        print(f'Processing P={P} ({p+1}/{nP})')

        # Select random fourier features (generated above as W)
        P_floor = int(np.floor(P / 2))    # calculate the largest integer less than or equal to half of P
        wtmp = W[:P_floor, :]  # get the random gaussian weights
        # matrix Z contains the random fourier features - using the gamma parameter
        Z = np.vstack([np.cos(gamma * wtmp.dot(X)), np.sin(gamma * wtmp.dot(X))])

        # out of sample loop to calculate predictions for each lambda shrinkage value
        # T are the total number of observations and trnwin is the training window
        for t in range(trnwin + 1, T + 1):

            trnloc = slice(t - trnwin - 1, t - 1) # calculate rolling training slice indices
            Ztrn = Z[:, trnloc] # get the training slice of random fourier features
            Ytrn = Y[trnloc] # get the training slice of outcome variables
            
            if demean:
                Ymn, Zmn = np.mean(Ytrn), np.mean(Ztrn, axis=1, keepdims=True)
            else:
                Ymn, Zmn = 0, 0
            
            Ytrn -= Ymn
            Ztrn -= Zmn
            Ztst = Z[:, t - 1] - Zmn
            Ztst = Ztst.reshape(-1, 1)  # Reshape Ztst to be two-dimensional
            
            Zstd = np.std(Ztrn, axis=1, keepdims=True)
            Ztrn /= Zstd
            Ztst /= Zstd
            
            if P <= trnwin: # run regular ridge regression
                # returns a set of coefficients for each lambda
                B = calculate_betas_using_standard_ridge( Y = Ytrn, 
                                                          X = Ztrn.T, 
                                                          lambda_list = lamlist)
            else:  # number of parameters strictly larger than observations, run adjusted ridge regression 
                B = calculate_betas_using_Kelly_method(   Y = Ytrn, 
                                                          X = Ztrn.T, 
                                                          lambda_list = lamlist)
            
            prediction_ = B.T.dot(Ztst) + Ymn # calculate prediction based on betas and mean (for each lambda shrinkage value)
            sum_squared_weights_ = np.sum(B ** 2, axis=0) # sum of squared weights
            
            Yprd[t - 1, p, :, 0] = prediction_.flatten()  # t is the time slice index, p is the index of the RFF number of parameters
            Bnrm[t - 1, p, :, 0] = sum_squared_weights_ # sum of squared weights;     t is the time slice index, p is the index of the RFF number of parameters
    
    #if rescale_predictions:
    # Reshape Y_scale_std to match Yprd's dimensions for broadcasting
    Y_scale_std_reshaped = Y_scale_std.reshape(Y_scale_std.shape[0], 1, 1, 1)
    Yprd_rescaled = Yprd * Y_scale_std_reshaped

    print(f"Total runtime: {time() - start_time} seconds")
    return Yprd, Yprd_rescaled, Bnrm


def benchmark_sim(X, Y, trnwin = 12, stdize = True):
    """
    Computes OOS forecasts using benchmark data, without RFF adjustment.  
        Applies Ridge regression with several shrinkage parameters. 
    
    Parameters:
    X : numpy.array
        M<ulti-dimensional numpy array of predictor variabes, Shape T x d with T observations of d variables.
    Y : numpy.array
        Outcome variable.   Shape T x 1.
    trnwin : int
        Training window size.
    stdize : bool
        Standardization flag. True to apply variance standardization of X and Y.
    rescale_predictions : bool
        scale predictions back to original scale
    """

    start_time = time()
    lamlist = 10.0 ** np.arange(-3, 4)  # list of lambdas for ridge regression
    demean = False # don't demean the variables in the out of sample run
    nL = len(lamlist)  # different lambdas for ridge

    # Add lag return (Y variable) as a predictor
    X = np.hstack([X, np.roll(Y, 1)])  ## this creates an issue becuase it moves the last value to the first
    X[0,(X.shape[1]-1)] = 0

    # Vol-standardize
    if stdize:
        # standardize X and Y by past 12 months' volatility, 
        #   computed as the square root of the average of the squared lags up to 12 months back.
        X, _ = volstdbwd(X)  # standardize X # uses expanding as default   
        Y, Y_scale_std = volstdbwd(Y, window=12)  # standardize Y  
        
        # Drop first 3 years due to vol scaling of X
        Y = Y[36:]
        Y_scale_std = Y_scale_std[36:] # discard the scaling factors for the first three years as well
        X = X[36:, :]
        #dates = dates[36:]

    T, d = X.shape # T is number of observations and d is number of variables
    X = X.T # transpose X
    Y = Y.reshape(-1, 1) # ensure Y is a two-dimensional array with one column (this is superfluous)

    # Empty Output Space based on number of Observations (T) RFF parameters (nP), number of lambdas (nL), number of simulations (nSim)
    Yprd = np.full((T, nL), np.nan)
    Bnrm = np.full((T, nL), np.nan)

    # out of sample loop to calculate predictions for each lambda shrinkage value
    # T are the total number of observations and trnwin is the training window
    for t in range(trnwin + 1, T + 1):

        trnloc = slice(t - trnwin - 1, t - 1) # calculate rolling training slice indices
        Ztrn = X[:, trnloc] # get the training slice of independent variables
        Ytrn = Y[trnloc] # get the training slice of outcome variables
        
        if demean:
            Ymn, Zmn = np.mean(Ytrn), np.mean(Ztrn, axis=1, keepdims=True)
        else:
            Ymn, Zmn = 0, 0
        
        Ytrn -= Ymn
        Ztrn -= Zmn
        Ztst = X[:, t - 1] - Zmn
        Ztst = Ztst.reshape(-1, 1)  # Reshape Ztst to be two-dimensional
        
        Zstd = np.std(Ztrn, axis=1, keepdims=True, ddof=1)  # this standardizes across rows, since X-matrix was transposed earlier
        Ztrn /= Zstd
        Ztst /= Zstd  
        
        #print(Ztrn)

        # returns a set of coefficients for each lambda
    # B = calculate_betas_P_less_than_or_equal_T(    Y = Ytrn, 
        B = calculate_betas_using_standard_ridge(   Y = Ytrn, 
                                                    X = Ztrn.T, 
                                                    lambda_list = lamlist)
        
        #print(B)
        
        prediction_ = B.T.dot(Ztst) + Ymn # calculate prediction based on betas and mean (for each lambda shrinkage value)
        sum_squared_weights_ = np.sum(B ** 2, axis=0) # sum of squared weights
        
        Yprd[t - 1, :] = prediction_.flatten()  # t is the time slice index, p is the index of the RFF number of parameters
        Bnrm[t - 1, :] = sum_squared_weights_ # sum of squared weights;     t is the time slice index, p is the index of the RFF number of parameters

    # re-scale prediction by vector of standard deviations
    #if rescale_predictions:
    Y_scale_std_reshaped = Y_scale_std.reshape(Y_scale_std.shape[0], 1)
    Yprd_rescaled = Yprd * Y_scale_std_reshaped

    print(f"Total runtime: {time() - start_time} seconds")
    return Yprd, Yprd_rescaled, Bnrm



#### Functions for Results Analysis


from scipy.stats import skew
import statsmodels.api as sm

def get_fit_stats(predictions_, Y, timing_method = 'multiply'):
    """
    Computes fit stats for a series of predictions and actual values. 
    
    Parameters:
    predictions_ : numpy.array
        Numpy array of predictions for each period, Shape T x 1 with T periods of predictions.
    Y : numpy.array
        Outcome variable.   Shape T x 1.
    """

    # calculate market timing returns - multiplying predictions by actual returns to get timing involves 
    #   evaluating how well the model's predictions align with actual market movements and, specifically, 
    #   whether the predictions successfully capture the direction and magnitude of those movements
    # If the prediction and the actual return have the same sign (both positive or both negative), 
    #   their product will be positive, indicating a correct directional forecast. 
    #   Conversely, a negative product indicates a mismatch in direction.
    # The magnitude of the product reflects how strong the prediction was in relation to the actual return. 
    #   Larger absolute values indicate that not only was the direction correctly predicted, 
    #   but the predicted magnitude was also significant
    #  By calculating this product across all time periods, the resulting series (timing) can be 
    #   interpreted as "market timing returns." Positive values in this series suggest periods where 
    #   the model successfully predicted the direction of the market, contributing positively to performance. 
    #   Negative values suggest periods of incorrect predictions, detracting from performance.
    if timing_method == 'multiply':
        timing = ( predictions_ * Y.T ) . squeeze()
    elif timing_method == 'AllInLongOnly':
        timing =  np.where(predictions_ > 0, Y.T, 0).squeeze()
     
    #slope, intercept, r_value, p_value, std_err = linregress(Y.T, timing)
    
    # Calculate annualized Expected Returns, Volatility and Sharpe Ratio (SR)
    ER = ( np.nanmean(timing) * 12 )
    vol = ( np.nanstd(timing) * np.sqrt(12) )
    SR =  ER /  vol

    # out of sample R-square value
    R2oos = 1 - np.var(Y.T - predictions_) / np.var(Y.T)

    #information ratio and slpha t-stats
    model = sm.OLS(timing, sm.add_constant(Y)).fit()
    intercept, slope = model.params
    t_stat_intercept, t_stat_slope = model.tvalues
    residuals = model.resid
    IRvMkt = np.sqrt(12)*  intercept / np.std(residuals)

    # calculate information ratio manually - 
   # excess_returns = timing - Y # excess returns (portfolio returns over the market)
   # mean_excess_return = np.mean(excess_returns) #mean of excess returns
   # std_excess_return = np.std(excess_returns, ddof=1) # sample standard deviation of excess returns
   # IRvMkt = mean_excess_return / std_excess_return 

    # Average Return t-stat
    x_bar = np.mean(timing) #sample mean (average return)
    mu = 0  # hypothesized population mean
    s = np.std(timing, ddof=1)  # ddof=1 for sample standard deviation
    avg_return_tStat = (x_bar - mu) / (s / np.sqrt(len(timing)))

    # Here, we directly use predictions_ assuming they represent return rates
    #cumulative_returns = np.nancumprod(1 + timing) - 1
    #  # Calculate cumulative returns
    #  cumulative_returns = (1 + timing).cumprod()

    #  # Calculate maximum drawdown
    #  peaks = np.maximum.accumulate(cumulative_returns)
    #  drawdowns = (cumulative_returns - peaks) / peaks
    #  max_drawdown = drawdowns.min()

    return {
        'R2oos'                 : R2oos,
        'ExpectedReturns'       : ER,
        'Volatility'            : vol,
        'SharpeRatio'           : SR,
        'Average Return'        : x_bar,
        'TStatAvgReturn'        : avg_return_tStat, #t_stat_intercept, #t_stat_slope,
        'IRvMkt'                : IRvMkt,
        'Alpha'                 : intercept,
        'AlphaTStat'            : t_stat_intercept,
        'MaxSinglePeriodLoss'   : min(timing),
        'Skew'                  : skew(timing),
         #'MaxDrawdown'       : max_drawdown,
      #  'CumulativeReturns': cumulative_returns[-1],  # Return the final cumulative return
    }

def calculate_real_investment_performance(unstandardized_predictions, unstandardized_Y, risk_aversion = 3):
    returns_forecast = unstandardized_predictions  
    #risk_aversion = 3
    _, estimated_variance = volstdbwd(unstandardized_Y.reshape(-1, 1,2), window=60)
    estimated_variance = estimated_variance.flatten() 
    # calculate weights
    weights = returns_forecast / ( risk_aversion * ( estimated_variance **2 ) )
    # Truncate weights at -1 and 2
    weights_truncated = np.clip(weights, -1, 2)
    # calculate cumulative returns
    timing = ( weights_truncated * unstandardized_Y[-len(weights_truncated):].flatten() ) 
    cumulative_returns = np.cumprod(1 + timing) - 1
    benchmark_cumulative_returns = np.cumprod(1 + unstandardized_Y) -1
    utility_gain = ( timing.mean() - 0.5 * risk_aversion * timing.var() ) - \
                   ( unstandardized_Y.mean() - 0.5 * risk_aversion * unstandardized_Y.var() )
    
    peaks = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peaks) / peaks
    max_drawdown = drawdowns.min()

    return weights_truncated, cumulative_returns, benchmark_cumulative_returns, utility_gain, max_drawdown

def get_real_investment_returns_dict(unstandardized_predictions, unstandardized_Y, risk_aversion = 3):
    weights_truncated, cumulative_returns, benchmark_cumulative_returns, utility_gain, max_drawdown =  \
            calculate_real_investment_performance(unstandardized_predictions = unstandardized_predictions, 
                                                  unstandardized_Y = unstandardized_Y,
                                                  risk_aversion = risk_aversion )
    return {
        "CumulativeReturn" : cumulative_returns[-1],
        "UtilityGain" : utility_gain,
        "MaxDrawDown" : max_drawdown,
        "BuyAndHoldCumulativeReturn" : benchmark_cumulative_returns[-1],
    }