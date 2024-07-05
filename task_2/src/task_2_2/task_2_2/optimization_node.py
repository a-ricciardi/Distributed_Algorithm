import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import networkx as nx
import os

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

def generate_targets(num_targets, radius):

    angles = np.linspace(0, 2 * np.pi, num_targets, endpoint=False)
    targets = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
    
    return targets

def generate_robots(num_robots, radius):

    robots = np.zeros((num_robots, 2))
    for i in range(num_robots):
        r = 0.5 * radius * np.sqrt(np.random.rand())  # sqrt to ensure uniform distribution in the circle
        theta = 2 * np.pi * np.random.rand()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        robots[i] = np.array([x, y])

    return robots

def cost_function(z_i, s_i, r_i, gamma_tar_i, gamma_agg_i):

    term1 = gamma_tar_i * np.linalg.norm(z_i - r_i)**2
    term2 = gamma_agg_i * np.linalg.norm(z_i - s_i)**2

    return term1 + term2

def cost_gradient(z_i, s_i, r_i, gamma_tar_i, gamma_agg_i):

    gradient_1 = 2 * gamma_tar_i * (z_i - r_i) + 2 * gamma_agg_i * (z_i - s_i)
    gradient_2 = 2 * gamma_agg_i * (s_i - z_i)

    return gradient_1, gradient_2

def phi(z):

    return z

class OptimizationNode(Node):

    def __init__(self):

        super().__init__('optimization_node')
        
        self.declare_parameter('num_agents', 6)
        self.declare_parameter('alpha', 0.001)
        self.declare_parameter('iterations', 1000)
        self.declare_parameter('random_seed', 42)
        self.declare_parameter('x_max', 5.0)
        self.declare_parameter('x_min', 15.0)
        self.declare_parameter('y_range', [-2.0, 2.0])
        self.declare_parameter('radius', 5.0)

        self.num_agents = self.get_parameter('num_agents').value
        self.alpha = self.get_parameter('alpha').value
        self.iterations = self.get_parameter('iterations').value
        
        #self.random_seed = self.get_parameter('random_seed').value
        #np.random.seed(self.random_seed)

        self.radius = self.get_parameter('radius').value
        self.robot_positions = generate_robots(self.num_agents, self.radius)
        self.rr = generate_targets(self.num_agents, self.radius)
        self.gamma_target = [5 for _ in range(self.num_agents)]
        self.gamma_aggregate = [5 for _ in range(self.num_agents)]
        self.G = nx.cycle_graph(self.num_agents)
        self.AA = metropolis_hastings_weights(self.G)
        self.trajectories = np.zeros((self.iterations, self.num_agents, 2))

        self.pub_positions = self.create_publisher(Float64MultiArray, 'robot_positions', 10)
        self.pub_targets = self.create_publisher(Float64MultiArray, 'target_positions', 10)
        self.pub_barycenter = self.create_publisher(Float64MultiArray, 'barycenter_position', 10)
        self.pub_trajectories = self.create_publisher(Float64MultiArray, 'robot_trajectories', 10)
       
        self.k = 0

        self.z = np.array(self.robot_positions)
        self.s = np.array([phi(self.z[i]) for i in range(self.num_agents)])
        self.v = np.zeros_like(self.z)
        for i in range(self.num_agents):
            _, gradient_2 = cost_gradient(self.z[i], self.s[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i])
            self.v[i] = gradient_2
        
        self.costs = []
        self.gradients = []
        self.robots = np.zeros((self.iterations, self.num_agents, 2))
        self.targets = np.zeros((self.iterations, self.num_agents, 2))
        self.estimates = np.zeros((self.iterations, self.num_agents, 2))

        # File paths for storing data
        self.data_dir = os.path.join(os.getcwd(), 'src/task_2_2/scripts')
        os.makedirs(self.data_dir, exist_ok=True)
        self.costs_file = os.path.join(self.data_dir, 'costs.txt')
        self.gradients_file = os.path.join(self.data_dir, 'gradients.txt')
        self.consensus_v_file = os.path.join(self.data_dir, 'consensus_v.txt')
        self.consensus_s_file = os.path.join(self.data_dir, 'consensus_s.txt')

        # Clear files if they exist
        open(self.costs_file, 'w').close()
        open(self.gradients_file, 'w').close()
        open(self.consensus_v_file, 'w').close()
        open(self.consensus_s_file, 'w').close()

        self.create_timer(1.0, self.aggregative_tracking_optimization)

    def aggregative_tracking_optimization(self):
        
        if self.k >= self.iterations:
            self.timer.cancel()
            return
            
        NN = self.num_agents
        alpha = self.alpha
        AA = self.AA

        z_new = np.copy(self.z)
        s_new = np.copy(self.s)
        v_new = np.copy(self.v)

        total_cost = 0
        gradient_1_total = 0
        gradient_2_total = 0
        gradient_2_new_total = []

        for i in range(NN):
            gradient_1, gradient_2 = cost_gradient(self.z[i], self.s[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i])
            z_new[i] = self.z[i] - alpha * (gradient_1 + np.dot(np.eye(2), self.v[i]))
            s_new[i] = sum(AA[i, j] * self.s[j] for j in range(NN)) + phi(z_new[i]) - phi(self.z[i])
            gradient_1_new, gradient_2_new = cost_gradient(z_new[i], s_new[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i])
            v_new[i] = sum(AA[i, j] * self.v[j] for j in range(NN)) + gradient_2_new - gradient_2
            gradient_2_new_total.append(gradient_2_new)
            gradient_1_total += gradient_1_new
            gradient_2_total += gradient_2_new
            cost = cost_function(self.z[i], self.s[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i])
            total_cost += cost

        self.z, self.s, self.v = z_new, s_new, v_new
        self.estimates[self.k] = self.s.copy()

        self.costs.append(total_cost)

        gradient_total = np.concatenate([gradient_1_total, gradient_2_total])
        self.gradients.append(gradient_total)

        self.robots[self.k] = self.z.copy()
        self.targets[self.k] = np.array(self.rr)
        self.trajectories[self.k] = self.z.copy()

        sigma = np.mean(self.z, axis=0)
        sum_gradient_2_new = np.sum(gradient_2_new_total, axis=0)

        # Save costs and gradients to file
        with open(self.costs_file, 'a') as f:
            f.write(f'{total_cost}\n')

        with open(self.gradients_file, 'a') as f:
            f.write(f'{np.linalg.norm(gradient_total)}\n')

        with open(self.consensus_v_file, 'a') as f:
            for i in range(self.num_agents):
                f.write(f'{np.linalg.norm(self.v[i] - sum_gradient_2_new / NN)} ')
            f.write('\n')

        with open(self.consensus_s_file, 'a') as f:
            for i in range(self.num_agents):
                f.write(f'{np.linalg.norm(self.s[i] - sigma)} ')
            f.write('\n')

        # Publish updated robot positions
        msg_positions = Float64MultiArray()
        msg_positions.data = self.z.flatten().tolist()
        self.pub_positions.publish(msg_positions)

        # Publish target positions
        msg_targets = Float64MultiArray()
        msg_targets.data = np.array(self.rr).flatten().tolist()
        self.pub_targets.publish(msg_targets)

        # Publish barycenter estimate position
        barycenter = np.mean(self.s, axis=0)
        msg_barycenter = Float64MultiArray()
        msg_barycenter.data = barycenter.flatten().tolist()
        self.pub_barycenter.publish(msg_barycenter)

        # Publish trajectories
        msg_trajectories = Float64MultiArray()
        msg_trajectories.data = self.trajectories[:self.k+1].flatten().tolist()
        self.pub_trajectories.publish(msg_trajectories)

        # if self.k % 100 == 0:
        #     self.get_logger().info(f'Iteration {self.k}: Published positions: {msg_positions.data}')
        #     self.get_logger().info(f'Iteration {self.k}: Total cost: {total_cost}')
        #     self.get_logger().info(f'Iteration {self.k}: Total gradient norm: {np.linalg.norm(np.concatenate([gradient_1_total, gradient_2_total]))}')
        #     self.get_logger().info(f'Iteration {self.k}: Published barycenter: {msg_barycenter.data}')
        #     self.get_logger().info(f'Iteration {self.k}: Published trajectories: {msg_trajectories.data}')

        self.k += 1
    

def main(args=None):
    rclpy.init(args=args)
    node = OptimizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()