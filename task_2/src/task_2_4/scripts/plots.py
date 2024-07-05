import matplotlib.pyplot as plt
import os

# Define the directory where the data files are stored
data_dir = os.path.join(os.getcwd(), 'src/task_2_2/scripts')
costs_file = os.path.join(data_dir, 'costs.txt')
gradients_file = os.path.join(data_dir, 'gradients.txt')
consensus_s_file = os.path.join(data_dir, 'consensus_s.txt')
consensus_v_file = os.path.join(data_dir, 'consensus_v.txt')

# Read cost data
with open(costs_file, 'r') as f:
    costs = [float(line.strip()) for line in f.readlines()]

# Read gradient norm data
with open(gradients_file, 'r') as f:
    gradients = [float(line.strip()) for line in f.readlines()]

# Read consensus s data
consensus_s = []
with open(consensus_s_file, 'r') as f:
    for line in f:
        consensus_s.append([float(val) for val in line.strip().split()])

consensus_s = list(zip(*consensus_s))  # Transpose to get individual agent data

# Read consensus v data
consensus_v = []
with open(consensus_v_file, 'r') as f:
    for line in f:
        consensus_v.append([float(val) for val in line.strip().split()])

consensus_v = list(zip(*consensus_v))  # Transpose to get individual agent data

# Plot cost function evolution
plt.figure(figsize=(6, 5))
plt.semilogy(costs, label='Cost Function')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Evolution')
plt.legend()
plt.grid(True)

plt.show()

# Plot gradient norm evolution
plt.figure(figsize=(6, 5))
plt.semilogy(gradients, label='Gradient Norm')
plt.xlabel('Iterations')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm Evolution')
plt.legend()
plt.grid(True)

plt.show()

# Plot consensus v evolution for each agent
plt.figure(figsize=(6, 5))
for i, agent_data in enumerate(consensus_v):
    plt.loglog(agent_data, label=f'Agent {i}')
plt.xlabel('Iterations')
plt.ylabel(r'$||v_i - \nabla_2 \ell (z_i, s_i)||$')
plt.title('Consensus on V')
plt.legend()
plt.grid(True)

plt.show()

# Plot consensus s evolution for each agent
plt.figure(figsize=(6, 5))
for i, agent_data in enumerate(consensus_s):
    plt.loglog(agent_data, label=f'Agent {i}')
plt.xlabel('Iterations')
plt.ylabel(r'$||s_i - \sigma(z)||$')
plt.title('Consensus on S')
plt.legend()
plt.grid(True)

plt.show()