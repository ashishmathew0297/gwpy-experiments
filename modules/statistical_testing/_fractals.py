import numba
import numpy as np

@numba.njit("f4[:](f4[:,:], i8, i8)", parallel=True)
def faster_compute_fractals_var(fs, dec, step):
    FDs = np.empty(shape=fs.shape[1], dtype=np.float32)
    N = fs.shape[1]
    for ii in numba.prange(fs.shape[0]):
        f = fs[ii]
        k_n = np.arange(1, N//(2*dec), step, np.int64)
        n_max = len(k_n)
        V_i = np.empty(shape=(n_max), dtype=np.float32)

        ub = np.empty(shape=(N-2*k_n[0],2), dtype=np.float32) # current iteration
        for i in range(0, N-2*k_n[0]):
            ub[i,0] = np.max(f[i:i+2*k_n[0]+1])
            ub[i,1] = np.min(f[i:i+2*k_n[0]+1])
        V_i[0] = np.mean(ub[:,0]-ub[:,1])

        for n in range(1,n_max):
            d = k_n[n] - k_n[n-1]
            for i in range(0, N-2*k_n[n]):
                ub[i,0] = max(ub[i,0], ub[i+2*d,0])
                ub[i,1] = min(ub[i,1], ub[i+2*d,1])
            V_i[n] = np.mean(ub[:N-2*k_n[n],0]-ub[:N-2*k_n[n],1])

        X = np.log(k_n)
        X_m = X - np.mean(X)
        Y = np.log(V_i)
        Y_m = Y - np.mean(Y)
        FDs[ii] = 2 - np.sum((X_m)*(Y_m))/np.sum((X_m)**2)
    return FDs