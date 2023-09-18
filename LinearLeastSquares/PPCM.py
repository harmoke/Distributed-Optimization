import numpy as np
import socket
import multiprocessing
import threading
import time, sys
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from scipy.linalg import *

def gradient(B, b, x):
    return B.T.dot(B.dot(x) - b)

def PPCM_GS(node_id, neighbors, N, B, b, s, eta, tol):
    # parameter settings
    node_port = PORT_MAP[node_id]
    step = 0
    err = 1
    beta = 1
    Ni = len(neighbors)
    # r : received; c : correction; p : prediction; l : lambda
    r_x_c = defaultdict(dict)
    r_x_p = defaultdict(dict)
    r_l = defaultdict(dict)
    # zero initial values
    dim = B.shape[1]
    x_c = np.zeros(dim)
    l = np.zeros(dim)
#   x_c = np.random.randn(dim)
#   l = np.random.randn(dim)
    x_p = np.zeros(dim)
    pos_x_c = {id : 0 for id in neighbors} # position for each node
    pos_x_p = {id : 0 for id in neighbors}
    pos_l = {id : 0 for id in neighbors}
    cnt = [0] # counter for nodes that are done
    res = [] # result of each step
    
    # thread for receiving
    t = threading.Thread(target=receive_data, args=(s, node_port, r_x_c, r_x_p, r_l, pos_x_c, pos_x_p, pos_l, cnt))
    t.start()
    
    while err > tol:
        # prediction step
        msg = pickle.dumps([step, 'x', 'c', x_c])
        
        for neighbor_id in neighbors:
            neighbor_port = PORT_MAP[neighbor_id]
            send_data(s, msg, node_port, neighbor_port)
        
        while len(r_x_c[step]) + cnt[0] < Ni:
            continue
        
        s_x_c = np.zeros(dim)
        for i in neighbors:
            s_x_c += r_x_c[min(step, pos_x_c[i])][i]
        
        msg = pickle.dumps([step, 'l', l])
        
        for neighbor_id in neighbors:
            neighbor_port = PORT_MAP[neighbor_id]
            send_data(s, msg, node_port, neighbor_port)
            
        while len(r_l[step]) + cnt[0] < Ni:
            continue
        
        s_l = np.zeros(dim)
        for i in neighbors:
            s_l += r_l[min(step, pos_l[i])][i]
        
        g_c = gradient(B, b, x_c)
        r = 0 # ratio for criteria
        while True:
            x_p = x_c - beta * g_c + beta * (Ni * l - s_l) / (N * 2)
            
            d_x = x_c - x_p # direction for x
            d_g = beta * (g_c - gradient(B, b, x_p))
            
            tmp_a = np.inner(d_g, d_g)
            tmp_b = np.inner(d_x, d_x)
            r = np.sqrt(tmp_a / tmp_b)
            
            if r > eta:
#               beta = beta * 0.8 / r
                beta = beta * (2.0 / 3.0) * min(1.0, 1.0 / r)
            else:
                break
        
        msg = pickle.dumps([step, 'x', 'p', x_p])
        
        for neighbor_id in neighbors:
            neighbor_port = PORT_MAP[neighbor_id]
            send_data(s, msg, node_port, neighbor_port)
        
        while len(r_x_p[step]) + cnt[0] < Ni:
            continue
        
        s_x_p = np.zeros(dim)
        for i in neighbors:
            s_x_p += r_x_p[min(step, pos_x_p[i])][i]
        
        l_pre = l + np.zeros(dim)
        l = l - (eta ** 2) / beta * (Ni * x_p - s_x_p) / (N * 2)
        
        msg = pickle.dumps([step + 1, 'l', l])
        
        for neighbor_id in neighbors:
            neighbor_port = PORT_MAP[neighbor_id]
            send_data(s, msg, node_port, neighbor_port)
            
        while len(r_l[step + 1]) + cnt[0] < Ni:
            continue
        
        s_l = np.zeros(dim)
        for i in neighbors:
            s_l += r_l[min(step + 1, pos_l[i])][i]
        
        # correction step
        x_c = x_c - beta * gradient(B, b, x_p) + beta * (Ni * l - s_l) / (N * 2)
        
        # adjust beta
        if r <= 0.5:
            beta = beta * 0.7 / r
#           beta = beta * 1.5
        
        # compute stepwise error
        err = max(np.linalg.norm(x_c - x_p, np.inf), np.linalg.norm(l - l_pre, np.inf))
        if step % 10 == 0:
            print(f"stepwise error of node {node_id} at step {step}: {err}")
        
        res.append(x_c)
        step += 1
    
    # d : done
    msg = pickle.dumps([step, 'd'])
    
    for neighbor_id in neighbors:
        neighbor_port = PORT_MAP[neighbor_id]
        send_data(s, msg, node_port, neighbor_port)
    
    t.join()
    s.close()
    return res

def send_data(s, msg, node_port, neighbor_port):
    s.sendto(msg, ("localhost", neighbor_port))
#   m = pickle.loads(msg)
#   print(f"node port {node_port} send to node port {neighbor_port}: {m}")
    
def receive_data(s, node_port, r_x_c, r_x_p, r_l, pos_x_c, pos_x_p, pos_l, cnt):
    buffersize = 65536000
    s.settimeout(3)
    while True:
        try:
            data, addr = s.recvfrom(buffersize)
        except:
            data = None
        if data is not None:
            rec = pickle.loads(data)
            k = rec[0]
            m = PORT_MAP[addr[1]]
            if rec[1] == 'x':
                if rec[2] == 'c':
                    r_x_c[k][m] = rec[3]
                    pos_x_c[m] = k
                else:
                    r_x_p[k][m] = rec[3]
                    pos_x_p[m] = k
            elif rec[1] == 'l':
                r_l[k][m] = rec[2]
                pos_l[m] = k
            else:
                cnt[0] += 1
#           print(f"node port {node_port} received from node port {addr[1]}: {rec}")
        else:
            break
    
def worker_process(node_id, neighbors, N, B, b, s):
    print(f"Node {node_id} begins to work.")
    tol = 1e-3 # tolerance
    eta = 0.9 # Lipschitz factor: 0 < eta < 1
    x = PPCM_GS(node_id, neighbors, N, B, b, s, eta, tol)
    step = len(x)
    print(f"Node {node_id} is done with {step} steps.")
    return_dict[node_id] = x
    
def create_graph():
    G = nx.Graph()
    
    # Input number of vertices and edges
    num_vertices = int(input("Enter the number of vertices: "))
    num_edges = int(input("Enter the number of edges: "))

    # Add nodes to the graph
    for i in range(num_vertices):
        G.add_node(i)
        
    # Add edges to the graph based on user input
    print("Now, enter the edges (format: node1 node2):")
    for _ in range(num_edges):
        u, v = map(int, input().split())
        G.add_edge(u, v)
        
    # Draw and show the graph
#   nx.draw(G, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue')
#   plt.show()
        
    return G

def find_free_ports(num_ports):
    free_ports = []
    mapping = {}
    
    for _ in range(num_ports):
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.bind(('localhost', 0))
        
        _, port = temp_socket.getsockname()
        free_ports.append(port)
        
        temp_socket.close()
        
    for i, port in enumerate(free_ports):
        mapping[i] = port
        mapping[port] = i
        
    return mapping

if __name__ == '__main__':
    # set graph
#   G = create_graph()
    N_nodes = int(sys.argv[1])
#   G = nx.cycle_graph(N_nodes)
    G = nx.complete_graph(N_nodes)
#   G = nx.erdos_renyi_graph(N_nodes, 0.5, seed=329, directed=False)
    
    # find free ports
    PORT_MAP = find_free_ports(N_nodes)
#   print(PORT_MAP)
    
    # create sockets
    sockets = {}
    for node_id in G.nodes():
        sockets[node_id] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockets[node_id].setblocking(False)
        sockets[node_id].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sockets[node_id].bind(("localhost", PORT_MAP[node_id]))
        
#   # generate B and b
#   B = np.random.randn(60000, 3000)
#   b = np.random.randn(B.shape[0])

    # read B and b from files
    B = np.array(pickle.load(open("matrix_B.txt", "rb")))
    b = np.array(pickle.load(open("vector_b.txt", "rb")))
    
    # partition B and b
    B_block = np.vsplit(B, N_nodes)
    b_block = np.hsplit(b, N_nodes)
    
    # build processes
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for node_id in G.nodes():
        p = multiprocessing.Process(target=worker_process, args=(node_id, list(G.adj[node_id]), N_nodes, B_block[node_id], b_block[node_id], sockets[node_id]))
        processes.append(p)
    
    # run processes
    t_start = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    t_end = time.time()
    print("total run time: %f" %(t_end - t_start - 3))
    
    # run built-in method
#   t_start = time.time()
#   x_std = np.linalg.lstsq(B, b, rcond=None)[0]
#   t_end = time.time()
#   print("standard run time: %f" %(t_end - t_start))
    
    x_std = np.array(pickle.load(open("solution_x.txt", "rb")))
    
    avg_err_2 = 0.0
    for node_id in G.nodes():
        tmp = np.linalg.norm(x_std - return_dict[node_id][-1])
        avg_err_2 += tmp
        print(f"L_2 error of node {node_id}: {tmp}")
    avg_err_2 /= N_nodes
    print(f"Average L_2 error: {avg_err_2}")
        
    avg_err_inf = 0.0
    for node_id in G.nodes():
        tmp = np.linalg.norm(x_std - return_dict[node_id][-1], np.inf)
        avg_err_inf += tmp
        print(f"L_inf error of node {node_id}: {tmp}")
    avg_err_inf /= N_nodes
    print(f"Average L_inf error: {avg_err_inf}")