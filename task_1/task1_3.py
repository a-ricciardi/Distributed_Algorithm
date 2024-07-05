import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def generate_dataset(M, d):

    return np.random.randn(M, d)

def split_dataset(dataset_size, N):

    indices = np.random.permutation(dataset_size)

    return np.array_split(indices, N)

def phi_linear(D):

    return D

def phi_ellipse(D):

    return np.array([D[0], D[1], D[0]**2, D[1]**2])

def label_points(dataset, w, b, phi):

    labels = []
    for D in dataset:
        transformed_D = phi(D)
        if np.dot(w, transformed_D) + b >= 0:
            pm = 1 
        else:
            pm = -1
        labels.append(pm)

    return np.array(labels)

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

def cost_function_and_gradient(z, dataset, labels, phi):

    w, b = z[:-1], z[-1]
    grad_w = np.zeros_like(w)
    grad_b = 0
    cost = 0
    M = dataset.shape[0]
    for i in range(M):
        D = dataset[i]
        label = labels[i]
        transformed_D = phi(D)
        linear_combination = np.dot(w, transformed_D) + b
        cost += np.log(1 + np.exp(-label * linear_combination))
        factor = -label / (1 + np.exp(label * linear_combination))
        grad_w += factor * transformed_D
        grad_b += factor

    return cost, np.concatenate([grad_w, [grad_b]])

def compute_misclassification_rate(original_labels, dataset, w, b, phi):

    new_labels = label_points(dataset, w, b, phi)
    misclassifications = np.sum(original_labels != new_labels)
    misclassification_rate = 100 * misclassifications / len(original_labels) 

    return misclassification_rate

def run_optimization(phi, q, label):

    print(f"Run optimization for {label} labeling...")

    # Split dataset into training and testing
    indices = np.random.permutation(M)
    train_indices = indices[:int(0.7 * M)]
    test_indices = indices[int(0.7 * M):]
    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]

    # Initialize weights and bias for labeling
    w = np.random.randn(q)
    b = np.random.randn()
    
    train_labels = label_points(train_dataset, w, b, phi)
    test_labels = label_points(test_dataset, w, b, phi)

    # Split dataset into training and test sets
    subsets_indices = split_dataset(len(train_dataset), NN)
    subsets = [train_dataset[subset_indices] for subset_indices in subsets_indices]
    labelsets = [train_labels[subset_indices] for subset_indices in subsets_indices]

    G = nx.cycle_graph(NN)
    AA = metropolis_hastings_weights(G)

    ZZ = np.zeros((NN, q+1))
    SS = np.zeros_like(ZZ)              # Initialize the gradient tracking matrix

    for ii in range(NN):
        SS[ii] = cost_function_and_gradient(ZZ[ii], subsets[ii], labelsets[ii], phi)[1]  # Initialize gradients

    costs = []
    gradient_norms = []
    gradient_list = []
    consensus_SS = np.zeros((iterations, NN))

    for kk in range(iterations):
        old_ZZ = np.copy(ZZ)
        
        old_gradients = np.array([cost_function_and_gradient(ZZ[ii], subsets[ii], labelsets[ii], phi)[1] for ii in range(NN)])
        ZZ = AA @ ZZ - alpha * SS
        new_gradients = np.array([cost_function_and_gradient(ZZ[ii], subsets[ii], labelsets[ii], phi)[1] for ii in range(NN)])
        SS = AA @ SS + new_gradients - old_gradients
        
        cost = np.sum([cost_function_and_gradient(ZZ[ii], subsets[ii], labelsets[ii], phi)[0] for ii in range(NN)])
        costs.append(cost)
        
        gradient_list.append((SS)) 
        gradient_norms = [np.linalg.norm(g) for g in gradient_list]

        gradient_sum = np.sum(new_gradients, axis=0)
        for ii in range(NN):
            consensus_SS[kk, ii] = np.linalg.norm(SS[ii] - gradient_sum / NN)

    ZZ_opt = np.mean(ZZ, axis=0)
    w_opt = ZZ_opt[:-1]
    b_opt = ZZ_opt[-1]

    # Evaluate on test set
    misclassification_rate_train = compute_misclassification_rate(train_labels, train_dataset, w_opt, b_opt, phi)
    misclassification_rate_test = compute_misclassification_rate(test_labels, test_dataset, w_opt, b_opt, phi)

    print(f"Results for {label}:")
    print("Cost Function:", cost)
    print("Misclassification Rate (Train) (%):", misclassification_rate_train)
    print("Misclassification Rate (Test) (%):", misclassification_rate_test)

    plt.figure(figsize=(6, 5))
    plt.semilogy(costs)
    plt.title(f'Cost Function Evolution ({label})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.semilogy(gradient_norms)
    plt.title(f'Gradient Norm Evolution ({label})')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 5))
    for ii in range(NN):
        plt.loglog(consensus_SS[:, ii], label=f'Agent {ii}')
    plt.title(f'Consensus on S ({label})')
    plt.xlabel('Iteration')
    plt.ylabel(r'$||s_i - \nabla \ell_i(z_i)||$')
    plt.legend()
    plt.grid(True)

    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(train_dataset[train_labels == 1][:, 0], train_dataset[train_labels == 1][:, 1], color='b', label='p^{m} = 1 (Train)')
    plt.scatter(train_dataset[train_labels == -1][:, 0], train_dataset[train_labels == -1][:, 1], color='r', label='p^{m} = -1 (Train)')
    plt.scatter(test_dataset[test_labels == 1][:, 0], test_dataset[test_labels == 1][:, 1], color='cyan', label='p^{m} = 1 (Test)', marker='x')
    plt.scatter(test_dataset[test_labels == -1][:, 0], test_dataset[test_labels == -1][:, 1], color='magenta', label='p^{m} = -1 (Test)', marker='x')

    x_vals = np.linspace(-3, 3, 1000)
    y_vals = np.linspace(-3, 3, 1000)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z_true = np.zeros_like(X_grid)
    Z_opt = np.zeros_like(X_grid)

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            D = np.array([X_grid[i, j], Y_grid[i, j]])
            transformed_D = phi(D)
            Z_true[i, j] = np.dot(w, transformed_D) + b
            Z_opt[i, j] = np.dot(w_opt, transformed_D) + b_opt

    plt.contour(X_grid, Y_grid, Z_true, levels=[0], colors='k')
    plt.contour(X_grid, Y_grid, Z_opt, levels=[0], colors='g')
    plt.plot([], [], 'k', label='True Decision Boundary')
    plt.plot([], [], 'g', label='Optimal Decision Boundary')
    plt.title(f'Dataset and Decision Boundary ({label})')
    plt.legend()
    plt.grid(True)
    
    plt.show()

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, NN))
    for subset_indices, color in zip(subsets_indices, colors):
        subset = train_dataset[subset_indices]
        plt.scatter(subset[:, 0], subset[:, 1], color=color, marker='o')
    for subset_indices, color in zip(subsets_indices, colors):
        test_indices_in_subset = np.intersect1d(test_indices, subset_indices)
        plt.scatter(test_dataset[np.isin(test_indices, test_indices_in_subset)][:, 0], test_dataset[np.isin(test_indices, test_indices_in_subset)][:, 1], color=color, alpha=0.7, marker='x')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='black', markersize=10),
                    Line2D([0], [0], marker='x', color='w', label='Test', markeredgecolor='black', markersize=10)]
    for i, color in enumerate(colors):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'Subset {i+1}', markerfacecolor=color, markersize=10))
    plt.title('Dataset Splitted into Subsets')
    plt.legend(handles=legend_elements, loc='best')
    plt.grid(True)
    
    plt.show()


# Setting the random seed for reproducibility
#np.random.seed(0)

# Parameters and initial setup
NN = 10                          # Number of agents
M = 1000                         # Total number of data points
d = 2                            # Dimensionality of data

iterations = 10000
alpha = 1e-3

dataset = generate_dataset(M, d)

# Run optimization for each transformation function
run_optimization(phi_linear, 2, "Linear")
run_optimization(phi_ellipse, 4, "Ellipse")