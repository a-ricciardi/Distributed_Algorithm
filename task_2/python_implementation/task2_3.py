import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def generate_targets(num_agents, x_min, y_range):

    x_targets = np.random.uniform(x_min, x_min + 5, num_agents)  # X > 15 and up to 25
    y_targets = np.random.uniform(y_range[0], y_range[1], num_agents)  # Near 0, e.g., -1 to 1

    return np.column_stack((x_targets, y_targets))

def generate_robots(num_agents, x_max, y_range):

    x_robots = np.random.uniform(x_max - 5, x_max, num_agents)  # X > 15 and up to 25
    y_robots = np.random.uniform(y_range[0], y_range[1], num_agents)  # Near 0, e.g., -1 to 1
    
    return np.column_stack((x_robots, y_robots))

def y_up(x, a, b, c):

    return (a*x + b)**8 + c

def y_low(x, a, b, c):

    return -((a*x + b)**8 + c)

def cost_function(z_i, s_i, r_i, gamma_target_i, gamma_aggregative_i, epsilon, a, b, c, t_obs=0.3):

    x_i, y_i = z_i

    g_up = y_i - y_up(x_i, a, b, c)
    g_low = -y_i + y_low(x_i, a, b, c)

    # High penalty if boundary conditions are violated
    if g_up < t_obs or g_low < t_obs:
        cost_barrier = 1e6  
    else:
        cost_barrier = -epsilon * (np.log(-g_up - t_obs) + np.log(-g_low - t_obs))

    cost_target = gamma_target_i * np.linalg.norm(z_i - r_i)**2
    cost_aggregative = gamma_aggregative_i * np.linalg.norm(z_i - s_i)**2

    return cost_target + cost_aggregative + cost_barrier

def compute_gradient(z_i, s_i, r_i, gamma_target_i, gamma_aggregative_i, epsilon, a, b, c, t_obs=0.3):
    
    x_i, y_i = z_i
    g_upper = y_i - y_up(x_i, a, b, c)
    g_lower = -y_i + y_low(x_i, a, b, c)

    # Calculate gradients
    du_dx = -8 * (a*x_i + b)**7 * a
    du_dy = 1
    dl_dx = 8 * (a*x_i + b)**7 * a
    dl_dy = -1

    grad_barrier_upper = np.array([du_dx, du_dy]) / (g_upper - t_obs)
    grad_barrier_lower = np.array([dl_dx, dl_dy]) / (g_lower - t_obs)
    grad_barrier = -epsilon * (grad_barrier_upper + grad_barrier_lower)

    grad_target = 2 * gamma_target_i * (z_i - r_i)
    grad_aggregative = 2 * gamma_aggregative_i * (s_i - z_i)

    return grad_target + grad_barrier, grad_aggregative

def phi(z):

    return z  

def aggregative_tracking_optimization(ZZ, rr, gamma_target, gamma_aggregative, AA, alpha, num_iterations, num_agents):

    print("Run optimization...")

    NN = num_agents
    SS = np.array([phi(ZZ[i]) for i in range(NN)])  # Initializing s_0
    VV = np.zeros_like(ZZ)  # Initialize v_0 with zero and then set properly
    
    # Calculate the initial v using only the barycenter part (grad_2)
    for ii in range(NN):
        _, gradient_2 = compute_gradient(ZZ[ii], SS[ii], rr[ii], gamma_target
    [ii], gamma_aggregative[ii], epsilon, a, b, c)
        VV[ii] = gradient_2  # Setting initial v_0
    
    costs = []
    gradients = []

    robots = np.zeros((num_iterations, num_agents, 2))          # Store robot positions for animation
    targets = np.zeros((num_iterations, num_agents, 2))         # Store target positions for animation
    estimates = np.zeros((num_iterations, num_agents, 2))
    consensus_VV = np.zeros((iterations, NN))
    consensus_SS = np.zeros((iterations, NN))

    for kk in range(num_iterations):
        ZZ_new = np.copy(ZZ)
        SS_new = np.copy(SS)
        VV_new = np.copy(VV)
        total_cost = 0
        gradient_1_total = 0
        gradient_2_total = 0

        for ii in range(NN):
            
            gradient_1, gradient_2 = compute_gradient(ZZ[ii], SS[ii], rr[ii], gamma_target
        [ii], gamma_aggregative[ii], epsilon, a, b, c)
            
            ZZ_new[ii] = ZZ[ii] - alpha * (gradient_1 + np.dot(np.eye(2), VV[ii]))
            SS_new[ii] = sum(AA[ii, j] * SS[j] for j in range(NN)) + phi(ZZ_new[ii]) - phi(ZZ[ii])

            gradient_1_new, gradient_2_new = compute_gradient(ZZ_new[ii], SS_new[ii], rr[ii], gamma_target
        [ii], gamma_aggregative[ii], epsilon, a, b, c)

            VV_new[ii] = sum(AA[ii, j] * VV[j] for j in range(NN)) + gradient_2_new - gradient_2

            sigma = np.mean(ZZ_new, axis=0)


            cost = cost_function(ZZ[ii], SS[ii], rr[ii], gamma_target
        [ii], gamma_aggregative[ii], epsilon, a, b, c)
            total_cost += cost

            # Sum of gradients
            gradient_1_total += gradient_1_new
            gradient_2_total += gradient_2_new

        ZZ, SS, VV = ZZ_new, SS_new, VV_new
        estimates[kk] = SS.copy()

        costs.append(total_cost)

        gradient_total = np.concatenate([gradient_1_total, gradient_2_total])
        gradients.append(gradient_total)  # Store total gradient norm per iteration
        
        robots[kk] = ZZ.copy()
        targets[kk] = np.array(rr)

        gradient_2_sum = gradient_2_total / NN
        for ii in range(NN):
            consensus_VV[kk, ii] = np.linalg.norm(VV[ii] - gradient_2_sum)
            consensus_SS[kk, ii] = np.linalg.norm(SS[ii] - sigma)

    return robots, targets, costs, gradients, estimates, consensus_VV, consensus_SS

def animate_robots(robots, targets, estimates, a, b, c):

    fig, ax = plt.subplots()
    ax.set_xlim([0, 20])
    ax.set_ylim([-3, 3])
    ax.set_title('Robots moving inside a corridor')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    # Generate x values for boundary plotting
    x_vals = np.linspace(np.min(ZZ[:, 0]) - 50, np.max(ZZ[:, 0]) + 50, 2000)  # Adjust range and increase density
    y_upper_vals = y_up(x_vals, a, b, c)
    y_lower_vals = y_low(x_vals, a, b, c)

    # Plotting the boundaries
    ax.fill_between(x_vals, y_upper_vals, y_lower_vals, facecolor='green', alpha=0.5, interpolate=True)
    ax.fill_between(x_vals, y_upper_vals, 3, facecolor='red', alpha=0.5, interpolate=True)
    ax.fill_between(x_vals, -3, y_lower_vals, facecolor='red', alpha=0.5, interpolate=True)
    ax.plot(x_vals, y_upper_vals, 'r--')
    ax.plot(x_vals, y_lower_vals, 'r--', label='Boundary')

    initial_scatter = ax.scatter(robots[0, :, 0], robots[0, :, 1], color='blue', alpha=0.5, label='Initial Positions')
    robot_scatter = ax.scatter([], [], color='blue', label='Robots')
    target_scatter = ax.scatter(targets[0, :, 0], targets[0, :, 1], color='red', marker='x', label='Targets')
    barycenter_scatter = ax.scatter([], [], color='yellow', marker='o', label='Estimated Barycenter')

    trajectories = [ax.plot([], [], 'b-', linewidth=0.5)[0] for _ in range(len(robots[0]))]
    ax.legend(loc='upper right')

    def update(frame):

        if frame >= len(estimates):
            return []
        robot_scatter.set_offsets(robots[frame])
        barycenter_scatter.set_offsets(estimates[frame])
        for i, line in enumerate(trajectories):
            line.set_data(robots[:frame + 1, i, 0], robots[:frame + 1, i, 1])
        return [robot_scatter, target_scatter, barycenter_scatter, initial_scatter] + trajectories

    ani = FuncAnimation(fig, update, frames=len(estimates), blit=True, interval=100)
    
    plt.grid(True)
    plt.show()


# Setting the random seed for reproducibility
#np.random.seed(0)

# Parameters
NN = 5
x_max = 5                       
x_min = 15                      
y_range_robot = (-1, 1)
y_range_target = (-2, 2)               
ZZ = generate_robots(NN, x_max, y_range_robot)
rr = generate_targets(NN, x_min, y_range_target)

gamma_target = [5 for _ in range(NN)]
gamma_aggregative = [5 for _ in range(NN)]
epsilon = 30
a = 0.3
b = -3
c = 0.5

alpha = 1e-3
iterations = 2000

G = nx.cycle_graph(NN)
AA = metropolis_hastings_weights(G)

robots, targets, cost_history, gradient_history, estimates, consensus_VV, consensus_SS = aggregative_tracking_optimization(ZZ, rr, gamma_target, gamma_aggregative, AA, alpha, iterations, NN)

gradient_norms = np.linalg.norm(gradient_history, axis=1)

# Plotting
fig1 = plt.figure(figsize=(6, 5))
plt.semilogy(cost_history)
plt.title('Cost Function Evolution')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)

plt.show()

fig2 = plt.figure(figsize=(6, 5))
plt.semilogy(gradient_norms)
plt.title('Gradient Norm Evolution')
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.grid(True)

plt.show()

fig3 = plt.figure(figsize=(6, 5))
for ii in range(NN):
    plt.loglog(consensus_VV[:, ii], label=f'Agent {ii}')
plt.title(f'Consensus on V')
plt.xlabel('Iteration')
plt.ylabel(r'$||v_i - \nabla_2 \ell_i (z_i, s_i)|$')
plt.legend()
plt.grid(True)

plt.show()

fig4 = plt.figure(figsize=(6, 5))
for ii in range(NN):
    plt.loglog(consensus_SS[:, ii], label=f'Agent {ii}')
plt.title(f'Consensus on S')
plt.xlabel('Iteration')
plt.ylabel(r'$||s_i - \sigma(z)||$')
plt.legend()
plt.grid(True)

plt.show()

# Animate robot movements
animate_robots(robots, targets, estimates[:, :, :2], a, b, c)