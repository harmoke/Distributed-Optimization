import numpy as np
import pickle, time, sys
from mpi4py import MPI

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def gradient(w, X, Y):
    res = np.zeros(dim, dtype = float)
    for i in range(M):
        z = np.inner(w, X[i])
        res += (sigmoid(z) - Y[i]) * X[i]
    return res / (M * N)

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
    N_i = len(neighbors)
    dim = len(X[0])
    # r : received
    x_r = {j : np.zeros(dim, dtype = float) for j in neighbors}
    # zero initial values
    x = np.zeros(dim, dtype = float)
    # done & results
    done = set() # set of finished nodes
    res = [] # result of each step
    
    # compute w_{ij}
    w_i = {j : 1 / N for j in neighbors}
    
    # WAGM
    t1 = time.time() # start time
    while err > tol:
        # exchange x
        send_requests = []
        recv_requests = []
        msg = x.copy()
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
                x_r[source] = message
        MPI.Request.Waitall(recv_requests)
        
        # wa : weighted average
        x_wa = np.zeros(dim, dtype = float)
        w_sum = 0
        for j in neighbors:
            x_wa += w_i[j] * x_r[j]
            w_sum += w_i[j]
        x_wa += (1.0 - w_sum) * x
        
        # update x
        x_pre = x.copy()
        alpha = 20.0 / (step + 1)
        x = x_wa - alpha * gradient(x_wa, X, Y)
        
        # compute stepwise error
        err = np.linalg.norm(x - x_pre, np.inf)
        if step % 10 == 0:
            print(f"stepwise L^inf error of node {id} at step {step} : {err}")
        
        # update result and step
        res.append(x)
        step += 1
    t2 = time.time() # end time
    
    # compute running time
    t = t2 - t1
    print(f"running time of node {id} : {t} seconds")
    
    # compute standard error
    x_std = np.array(pickle.load(open("Data/Sol/w_std.txt", "rb")))
    err_2 = np.linalg.norm(x - x_std)
    err_inf = np.linalg.norm(x - x_std, np.inf)
    print(f"L^2 error of node {id} : {err_2}")
    print(f"L^inf error of node {id} : {err_inf}")
    
    # save the result
    pickle.dump([t, res], open(f"Res/WAGM_N={N}_{type}_{id}.txt", "wb"))
    
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