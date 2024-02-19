import numpy as np
import pickle, time, sys
from mpi4py import MPI

def gradient(w, X, Y):
    res = np.zeros(dim, dtype = float)
    for i in range(M):
        z = 1.0 - Y[i] * np.inner(w, X[i])
        if z > 0:
            res -= Y[i] * X[i]
    res = 0.1 * res / (N * M)
    res += w / N
    return res

if __name__ == '__main__':
    # start communication
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    id = comm.Get_rank()
    N = comm.Get_size()
    
    # load data
    X = np.array(pickle.load(open(f"Data/N={N}/X_allocated_{id}.txt", "rb")))
    Y = np.array(pickle.load(open(f"Data/N={N}/Y_allocated_{id}.txt", "rb")))
    M = len(Y)
    type = sys.argv[1]
    neighbors = list(pickle.load(open(f"Graph/N={N}/{type}/neighbor_of_{id}.txt", "rb")))
    print(f"neighbors of node {id} : {neighbors}")
    
    # parameter setting
    step = 0
    err = 1.0
    tol = 1e-6
    eta = 0.9
    r_i = 1.0
    dim = len(X[0])
    # r : received; c : correction; p : prediction; l : lambda
    w_p_r = {j : np.zeros(dim, dtype = float) for j in neighbors}
    w_c_r = {j : np.zeros(dim, dtype = float) for j in neighbors}
    l_r = {j : np.zeros(dim, dtype = float) for j in neighbors}
    # zero initial values
    w_p = np.zeros(dim, dtype = float)
    w_c = np.zeros(dim, dtype = float)
    l = np.zeros(dim, dtype = float)
    l_sum = np.zeros(dim, dtype = float)
    # done & results
    done = set() # set of finished nodes
    res = [] # result of each step
    
    # get size of each neighbor
    N_i = len(neighbors)
    N_j = {j : 0 for j in neighbors}
    send_requests = []
    recv_requests = []
    msg = N_i
    for j in neighbors:
        req = comm.isend(msg, dest = j, tag = 0)
        send_requests.append(req)
    for j in neighbors:
        req = comm.irecv(source = j, tag = 0)
        recv_requests.append(req)
    MPI.Request.Waitall(send_requests)
    for _, req in enumerate(recv_requests):
        message = req.wait(status = status)
        source = status.Get_source()
        N_j[source] = message
    MPI.Request.Waitall(recv_requests)
    
    # compute a_{ij} & rho_i
    a_i = {j : 0.5 * eta / max(N_i, N_j[j]) for j in neighbors}
    rho_i = 0.0
    for j in neighbors:
        rho_i += a_i[j]
    
    # PPCM
    t1 = time.time() # start time
    while err > tol:
        # prediction for w
        mu = 0.0 # ratio for criteria
        g_c = gradient(w_c, X, Y)
        while True:
            w_p = w_c - (g_c - l_sum) / r_i
            w_p = np.maximum(0, w_p)
            
            d_w = (w_c - w_p) * r_i
            g_p = gradient(w_p, X, Y)
            d_g = g_c - g_p
            
            num_g = np.inner(d_g, d_g) # numerator
            den_w = np.inner(d_w, d_w) # denominator
            mu = np.sqrt(2.0 * num_g / den_w)
            
            # self-tuning
            if mu > eta:
                r_i = r_i * 1.5 * max(1.0, mu)
#               r_i = r_i * mu / 0.8
            else:
                break
        
        # exchange w_p
        send_requests = []
        recv_requests = []
        msg = w_p.copy()
        for j in neighbors:
            if j in done:
                continue
            req = comm.isend(msg, dest = j, tag = 1)
            send_requests.append(req)
        for j in neighbors:
            if j in done:
                continue
            req = comm.irecv(source = j, tag = 1)
            recv_requests.append(req)
        MPI.Request.Waitall(send_requests)
        for _, req in enumerate(recv_requests):
            message = req.wait(status = status)
            source = status.Get_source()
            if isinstance(message, str):
                done.add(source)
            else:
                w_p_r[source] = message
        MPI.Request.Waitall(recv_requests)
        
        # update lambda
        w_p_sum = np.zeros(dim, dtype = float)
        for j in neighbors:
            w_p_sum += a_i[j] * (w_p - w_p_r[j])
        l_pre = l.copy()
        s_i = (eta / rho_i * 0.5) ** 2 * r_i
        l = l - s_i * w_p_sum
        
        # exchange l
        send_requests = []
        recv_requests = []
        msg = l.copy()
        for j in neighbors:
            if j in done:
                continue
            req = comm.isend(msg, dest = j, tag = 2)
            send_requests.append(req)
        for j in neighbors:
            if j in done:
                continue
            req = comm.irecv(source = j, tag = 2)
            recv_requests.append(req)
        MPI.Request.Waitall(send_requests)
        for _, req in enumerate(recv_requests):
            message = req.wait(status = status)
            source = status.Get_source()
            if isinstance(message, str):
                done.add(source)
            else:
                l_r[source] = message
        MPI.Request.Waitall(recv_requests)
        
        # correction for w
        l_sum = np.zeros(dim, dtype = float)
        for j in neighbors:
            l_sum += a_i[j] * (l - l_r[j])
        w_c = w_c - (g_p - l_sum) / r_i
        w_c = np.maximum(0, w_c)
        
        # adjust r_i
        if mu <= 0.5:
#           r_i = r_i * mu / 0.7
            r_i = r_i / 1.5
        
        # compute stepwise error
        err = max(np.linalg.norm((w_c - w_p) * np.sqrt(r_i), np.inf), np.linalg.norm((l - l_pre) / np.sqrt(s_i), np.inf))
#       err = max(np.linalg.norm(w_c - w_p, np.inf), np.linalg.norm(l - l_pre, np.inf))
        if step % 100 == 0:
            print(f"stepwise L^inf error of node {id} at step {step} : {err}")
        
        # update result and step
        res.append(w_c)
        step += 1
    t2 = time.time() # end time
    
    # compute running time
    t = t2 - t1
    print(f"running time of node {id} : {t} seconds")
    
    # compute standard error
    w_std = np.array(pickle.load(open("Data/Sol/w_std.txt", "rb")))
    err_2 = np.linalg.norm(w_c - w_std)
    err_inf = np.linalg.norm(w_c - w_std, np.inf)
    print(f"L^2 error of node {id} : {err_2}")
    print(f"L^inf error of node {id} : {err_inf}")
    
    # save the result
    pickle.dump([t, res], open(f"Res/PPCM_N={N}_{type}_{id}.txt", "wb"))
    
    # done
    send_requests = []
    recv_requests = []
    msg = "done"
    for j in neighbors:
        if j in done:
            continue
        req = comm.isend(msg, dest = j, tag = 1)
        send_requests.append(req)
    for j in neighbors:
        if j in done:
            continue
        req = comm.irecv(source = j, tag = 1)
        recv_requests.append(req)
    MPI.Request.Waitall(send_requests)
    MPI.Request.Waitall(recv_requests)