import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(M, d):

    return np.random.randn(M, d)

def phi_ellipse(D):

    return np.array([D[0], D[1], D[0]**2, D[1]**2])

def phi_linear(D):

    return D

def label_points(dataset, w, b, phi):

    labels = []
    for D in dataset:
        transformed_D = phi(D)
        if w @ transformed_D + b >= 0:
            pm = 1 
        else: 
            pm = -1
        labels.append(pm)

    return np.array(labels)

def logistic_regression_function(w, b, dataset, labels, phi):

    M = dataset.shape[0]
    cost = 0
    for i in range(M):
        D = dataset[i]
        pm = labels[i]
        transformed_D = phi(D)
        linear_combination = w @ transformed_D + b
        cost += np.log(1 + np.exp(-pm * linear_combination))

    return cost 

def compute_gradient(w, b, dataset, labels, phi):

    M = dataset.shape[0]
    grad_w = np.zeros_like(w)
    grad_b = 0
    for ii in range(M):
        D = dataset[ii]
        pm = labels[ii]
        transformed_D = phi(D)
        linear_combination = w @ transformed_D + b
        exp_term = np.exp(-pm * linear_combination)
        grad_w += -pm * transformed_D * exp_term / (1 + exp_term)
        grad_b += -pm * exp_term / (1 + exp_term)

    return grad_w, grad_b

def centralized_gradient_descent(w, b, dataset, labels, alpha, iterations, phi):

    cost_history = []
    gradient_list = []
    for ii in range(iterations):
        grad_w, grad_b = compute_gradient(w, b, dataset, labels, phi)
        w -= alpha * grad_w
        b -= alpha * grad_b
        cost = logistic_regression_function(w, b, dataset, labels, phi)
        cost_history.append(cost)
        gradient_list.append(np.append(grad_w, grad_b))

    return w, b, cost_history, gradient_list

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

    w_opt, b_opt, cost_history, gradient_list = centralized_gradient_descent(w, b, train_dataset, train_labels, alpha, iterations, phi)

    gradient_norms = [np.linalg.norm(g) for g in gradient_list]
    
    # Evaluate misclassification rates
    train_misclassification_rate = compute_misclassification_rate(train_labels, train_dataset, w_opt, b_opt, phi)
    test_misclassification_rate = compute_misclassification_rate(test_labels, test_dataset, w_opt, b_opt, phi)

    print(f"Results for {label}:")
    print("Cost:", cost_history[-1])
    print("Training Misclassification Rate (%):", train_misclassification_rate)
    print("Testing Misclassification Rate (%):", test_misclassification_rate)
    
    plt.figure(figsize=(6, 5))
    plt.semilogy(cost_history)
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

    # Plot decision boundaries and data points for training and testing sets
    plt.figure(figsize=(10, 6))
    plt.scatter(train_dataset[train_labels == 1][:, 0], train_dataset[train_labels == 1][:, 1], color='blue', label='Train: Label 1')
    plt.scatter(train_dataset[train_labels == -1][:, 0], train_dataset[train_labels == -1][:, 1], color='red', label='Train: Label -1')
    plt.scatter(test_dataset[test_labels == 1][:, 0], test_dataset[test_labels == 1][:, 1], color='cyan', marker='x', label='Test: Label 1')
    plt.scatter(test_dataset[test_labels == -1][:, 0], test_dataset[test_labels == -1][:, 1], color='magenta', marker='x', label='Test: Label -1')

    x_vals = np.linspace(-3, 3, 1000)
    y_vals = np.linspace(-3, 3, 1000)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    ZZ_true = np.zeros_like(X_grid)
    ZZ_opt = np.zeros_like(X_grid)

    for i in range(X_grid.shape[0]):
        for j in range(X_grid.shape[1]):
            D = np.array([X_grid[i, j], Y_grid[i, j]])
            transformed_D = phi(D)
            ZZ_true[i, j] = w @ transformed_D + b
            ZZ_opt[i, j] = w_opt @ transformed_D + b_opt

    plt.contour(X_grid, Y_grid, ZZ_true, levels=[0], colors='k')
    plt.contour(X_grid, Y_grid, ZZ_opt, levels=[0], colors='g')
    plt.plot([], [], 'k', label='True Decision Boundary')
    plt.plot([], [], 'g', label='Optimal Decision Boundary')

    plt.title(f'Dataset and Decision Boundaries ({label})')
    plt.legend()
    plt.grid(True)
    
    plt.show()


# Setting the random seed for reproducibility
#np.random.seed(0)

# Parameters and initial setup
M = 1000
d = 2

iterations = 5000
alpha = 1e-2

dataset = generate_dataset(M, d)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(dataset[:, 0], dataset[:, 1], alpha=1)
plt.title('Generated Dataset')
plt.grid(True)
plt.show()

# Run optimization for each transformation function
run_optimization(phi_linear, d, "linear")
run_optimization(phi_ellipse, 2*d, "ellipse")