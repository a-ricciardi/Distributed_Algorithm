import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def metropolis_hastings_weights(G):

    AA = np.zeros((len(G), len(G)))
    for ii in G.nodes():
        N_ii = list(G.neighbors(ii))
        deg_ii = len(N_ii)
        for jj in N_ii:
            deg_jj = len(list(G.neighbors(jj)))
            AA[ii, jj] = 1 / (1 + max(deg_ii, deg_jj))
        AA[ii, ii] = 1 - np.sum(AA[ii, :])
        
    return AA


# Setting the random seed for reproducibility
#np.random.seed(0)

# Parameters
NN = 10                         # Number of agents
d = 2                           # Dimension of the space
alpha = 1e-2                    # Stepsize for gradient descent
iterations = 10000               # Number of iterations

# Generate graphs
graphs = {
    'Cycle': nx.cycle_graph(NN),
    'Path': nx.path_graph(NN),
    'Star': nx.star_graph(NN - 1)
}

# Quadratic function coefficients (affine terms)
QQ = np.random.uniform(size=(NN, d, d))
R = np.random.uniform(size=(NN, d))

# print(f"R: {R}")
# print(f"Q: {QQ}")
# print(f"R: {R.shape}")
# print(f"Q: {QQ.shape}")

# Ensure Q is positive definite
for i in range(NN):
    QQ[i] = np.dot(QQ[i].T, QQ[i])  # Making Q positive definite

for graph_name, G in graphs.items():

    print(f"Running Gradient Tracking on {graph_name} Graph")

    # Check if the graph is completely connected
    if nx.is_connected(G) and G.number_of_edges() == NN * (NN - 1) // 2:
        raise ValueError(f"The {graph_name} graph is completely connected. Exiting...")
    
    # Metropolis-Hastings weights for the adjacency matrix
    AA = metropolis_hastings_weights(G)

    # Initialization
    ZZ = np.zeros((iterations, NN, d))
    SS = np.zeros((iterations, NN, d))
    cost = np.zeros(iterations)

    # Initial gradient
    for ii in range(NN):
        SS[0, ii] = QQ[ii] @ ZZ[0, ii] + R[ii]

    # Gradient Tracking Algorithm
    for kk in range(iterations - 1):
        for ii in range(NN):
            N_ii = list(G.neighbors(ii))
            
            ZZ[kk + 1, ii] = AA[ii, ii] * ZZ[kk, ii]
            SS[kk + 1, ii] = AA[ii, ii] * SS[kk, ii]
            # print(AA[ii, ii])

            for jj in N_ii:
                ZZ[kk + 1, ii] += AA[ii, jj] * ZZ[kk, jj]
                SS[kk + 1, ii] += AA[ii, jj] * SS[kk, jj]

            ZZ[kk + 1, ii] -= alpha * SS[kk, ii]

            # Computation of new and old gradients
            grad_ll_new = QQ[ii] @ ZZ[kk + 1, ii] + R[ii]
            grad_ll_old = QQ[ii] @ ZZ[kk, ii] + R[ii]
            SS[kk + 1, ii] += grad_ll_new - grad_ll_old

            # Compute cost
            ll = 0.5 * ZZ[kk, ii].T @ QQ[ii] @ ZZ[kk, ii] + R[ii] @ ZZ[kk, ii]
            
            cost[kk] += np.sum(ll)  # Summing the scalar costs for each agent
            #print(f"cost: {cost} at iteration {kk}")

    # Calculate the aggregated Q and R for simpler notation
    Q_sum = np.sum(QQ, axis=0)
    R_sum = np.sum(R, axis=0)

    # Calculate the optimal cost
    ZZ_opt = -np.linalg.inv(Q_sum) @ R_sum
    opt_cost = 0.5 * ZZ_opt.T @ Q_sum @ ZZ_opt + R_sum.T @ ZZ_opt
    
    # Calculate the norm of the gradient at each iteration
    norm_gradient = np.linalg.norm(SS, axis=(1, 2))

    # print(f"Optimal Cost for {graph_name}: {opt_cost}")
    # print(f"Final Cost for {graph_name}: {cost[-998]}")

    # Graph visualization
    fig1 = plt.figure(figsize=(6, 5))
    plt.title(f'{graph_name} Graph')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    plt.grid(True)

    plt.show()

    # Plot the cost evolution
    fig2 = plt.figure(figsize=(6, 5))
    plt.semilogy(np.arange(iterations - 1), np.abs(cost - opt_cost)[:-1])
    plt.title(f'Cost Function Evolution ({graph_name} Graph)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$|\ell(z_i) - \ell(z^*)|$')
    plt.grid(True)

    plt.show()

    # Norm of the gradient evolution
    fig3 = plt.figure(figsize=(6, 5))
    plt.semilogy(np.arange(iterations), norm_gradient)
    plt.title(f'Gradient Norm Evolution ({graph_name} Graph)')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.grid(True)

    plt.show()

    ZZ_mean = np.mean(ZZ, axis=1)
    fig4 = plt.figure(figsize=(6, 5))
    for ii in range(NN):
        deviations = np.linalg.norm(ZZ[:, ii, :] - ZZ_mean, axis=1)
        plt.loglog(np.arange(iterations), deviations, label=f'Agent {ii}')
    plt.title(f'Consensus on Z ({graph_name} Graph)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$||z_i - z_{mean}||$')
    plt.legend()
    plt.grid(True)

    plt.show()

    SS_mean = np.mean(SS, axis=1)
    fig5 = plt.figure(figsize=(6, 5))
    for ii in range(NN):
        deviations = np.linalg.norm(SS[:, ii, :] - SS_mean, axis=1)
        plt.loglog(np.arange(iterations), deviations, label=f'Agent {ii}')
    plt.title(f'Consensus on S ({graph_name} Graph)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$||s_i - s_{mean}||$')
    plt.legend()
    plt.grid(True)

    plt.show()