reg_args = {'vma0': 14.5,
 'ws1': -0.105,
 'sigma_w0': 0.223,
 'sigma_w1': 0.0003,
 'sigma_e': 0.03,
 'corr_12': 1,
 'scales': 'loglog',
 'reg_type': 'map'
 }

vma0 = reg_args['vma0'] 
ws1 = reg_args['ws1']
sigma_w0 = reg_args['sigma_w0']
sigma_w1 = reg_args['sigma_w1']
reg_args['wt1'] = 1-ws1
reg_args['wt0'] = -(np.log(vma0/3.6)-np.log(vma0*100)*ws1)
reg_args['cov_12'] = reg_args['corr_12']*(sigma_w0*sigma_w1)  
    
    
def fit_t(reg_dists, times, reg_args):
    from scipy.linalg import inv 
      
    wt0 = reg_args['wt0']
    wt1 = reg_args['wt1']
    sigma_w0 = reg_args['sigma_w0']
    sigma_w1 = reg_args['sigma_w1']
    cov_12 = reg_args['cov_12']  
    sigma_e = reg_args['sigma_e']        
    
    led = np.log(list(reg_dists))
    X = np.vstack([np.ones(len(led)), led]).T
    y = np.log(list(times))  
    
    W0 = np.array([wt0, wt1])
    cov_w = np.array([[sigma_w0, cov_12],[cov_12, sigma_w1]])
    cov_e = np.eye(len(led))*sigma_e
    
    W_map = W0 + inv( inv(cov_w) + X.T @ inv(cov_e) @ X) @ X.T @ inv(cov_e) @ (y - X @ W0)   
    W_ml = inv(X.T @ X) @ X.T @ y

    return W_map, W_ml
