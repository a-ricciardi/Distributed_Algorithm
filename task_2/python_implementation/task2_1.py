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

def cost_function(zz_i, ss_i, rr_i, gamma_target_i, gamma_aggregative_i):

    cost_target = gamma_target_i * np.linalg.norm(zz_i - rr_i)**2
    cost_aggregative = gamma_aggregative_i * np.linalg.norm(zz_i - ss_i)**2
    
    return cost_target + cost_aggregative

def compute_gradient(zz_i, ss_i, rr_i, gamma_i, gamma_aggregative_i):

    grad_1 = 2 * gamma_i * (zz_i - rr_i) + 2 * gamma_aggregative_i * (zz_i - ss_i)
    grad_2 = 2 * gamma_aggregative_i * (ss_i - zz_i)

    return grad_1, grad_2

def phi(z):

    return z  

def aggregative_tracking_optimization(ZZ, rr, gamma_target, gamma_aggregative, AA, alpha, iterations, num_agents):

    print("Run optimization...")

    NN = num_agents
    SS = np.array([phi(ZZ[i]) for i in range(NN)])  # Initializing s_0
    VV = np.zeros_like(ZZ)  # Initialize v_0 with zero and then set properly
    
    # Calculate the initial v using only the barycenter part (grad_2)
    for ii in range(NN):
        _, gradient_2 = compute_gradient(ZZ[ii], SS[ii], rr[ii], gamma_target[ii], gamma_aggregative[ii])
        VV[ii] = gradient_2  # Setting initial v_0
    
    costs = []
    gradients = []

    robots = np.zeros((iterations, num_agents, 2))          # Store robot positions for animation
    targets = np.zeros((iterations, num_agents, 2))         # Store target positions for animation
    estimates = np.zeros((iterations, num_agents, 2))
    consensus_VV = np.zeros((iterations, NN))
    consensus_SS = np.zeros((iterations, NN))

    for kk in range(iterations):
        ZZ_new = np.copy(ZZ)
        SS_new = np.copy(SS)
        VV_new = np.copy(VV)
        total_cost = 0
        gradient_1_total = 0
        gradient_2_total = 0

        for ii in range(NN):
            
            gradient_1, gradient_2 = compute_gradient(ZZ[ii], SS[ii], rr[ii], gamma_target[ii], gamma_aggregative[ii])
            
            ZZ_new[ii] = ZZ[ii] - alpha * (gradient_1 + np.dot(np.eye(2), VV[ii]))
            SS_new[ii] = sum(AA[ii, j] * SS[j] for j in range(NN)) + phi(ZZ_new[ii]) - phi(ZZ[ii])

            gradient_1_new, gradient_2_new = compute_gradient(ZZ_new[ii], SS_new[ii], rr[ii], gamma_target[ii], gamma_aggregative[ii])
            VV_new[ii] = sum(AA[ii, j] * VV[j] for j in range(NN)) + gradient_2_new - gradient_2

            sigma = np.mean(ZZ_new, axis=0)

            cost = cost_function(ZZ[ii], SS[ii], rr[ii], gamma_target[ii], gamma_aggregative[ii])
            total_cost += cost

            # Sum of gradients
            gradient_1_total += gradient_1_new
            gradient_2_total += gradient_2_new

        ZZ, SS, VV = ZZ_new, SS_new, VV_new
        estimates[kk] = np.mean(SS.copy())

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

def animate_robots(robots, targets, estimates):
    
    fig, ax = plt.subplots()
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_title('Robots moving in tight formation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    initial_scatter = ax.scatter(robots[0, :, 0], robots[0, :, 1], color='blue', alpha=0.25, label='Initial Positions')
    robot_scatter = ax.scatter([], [], color='blue', label='Robots')
    target_scatter = ax.scatter(targets[0, :, 0], targets[0, :, 1], color='red', marker='x', label='Targets')
    barycenter_scatter = ax.scatter([], [], color='orange', marker='o', label='Estimated Barycenter')

    trajectories = [ax.plot([], [], 'b-', linewidth=0.5)[0] for _ in range(len(robots[0]))]
    ax.legend(loc='upper right')

    def update(frame):
        if frame >= len(estimates):
            return []
        robot_scatter.set_offsets(robots[frame])
        barycenter_scatter.set_offsets(estimates[frame])
        for i, line in enumerate(trajectories):
            line.set_data(robots[:frame + 1, i, 0], robots[:frame + 1, i, 1])
        return [robot_scatter, target_scatter, initial_scatter, barycenter_scatter] + trajectories

    ani = FuncAnimation(fig, update, frames=len(estimates), blit=True, interval=100)
    
    plt.grid(True)
    plt.show()


# Setting the random seed for reproducibility
#np.random.seed(0)

# Parameters
NN = 6
ZZ = np.random.randn(NN, 2) * 0.1
radius = 3
rr = []

# Generate target positions on the circumference of the circle
angles = np.linspace(0, 2 * np.pi, NN, endpoint=False)
for angle in angles:
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    rr.append(np.array([x, y]))

# Generate robot positions randomly inside the circle
for i in range(NN):
    r = radius * np.sqrt(np.random.rand())  # sqrt to ensure uniform distribution in the circle
    theta = 2 * np.pi * np.random.rand()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ZZ[i] = np.array([x, y])

gamma_target = [3 for _ in range(NN)]
gamma_aggregative = [3 for _ in range(NN)]
alpha = 1e-3
iterations = 2000

G = nx.cycle_graph(NN)
AA = metropolis_hastings_weights(G)

# Run optimization
robots, targets, cost_history, gradient_history, estimates, consensus_VV, consensus_SS = aggregative_tracking_optimization(ZZ, rr, gamma_target, gamma_aggregative, AA, alpha, iterations, NN)

#print final cost
print(f"Final Cost: {cost_history[-1]}")

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
animate_robots(robots, targets, estimates[:, :, :2])