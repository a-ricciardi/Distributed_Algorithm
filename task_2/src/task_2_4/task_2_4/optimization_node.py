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

def generate_targets(num_targets, x_min, y_range):

    x_targets = np.random.uniform(x_min, x_min + 5, num_targets)
    y_targets = np.random.uniform(y_range[0], y_range[1], num_targets)
    
    return np.column_stack((x_targets, y_targets))

def generate_robots(num_robots, x_max, y_range):

    x_robots = np.random.uniform(x_max - 5, x_max, num_robots)
    y_robots = np.random.uniform(y_range[0], y_range[1], num_robots)
    
    return np.column_stack((x_robots, y_robots))

def y_upper(x, a, b, c):

    return (a * x + b)**8 + c

def y_lower(x, a, b, c):

    return -((a * x + b)**8 + c)

def cost_function(z_i, s_i, r_i, gamma_tar_i, gamma_agg_i, epsilon, a, b, c, t_obs=0.2):
    
    x_i, y_i = z_i
    g_upper = y_i - y_upper(x_i, a, b, c)
    g_lower = -y_i + y_lower(x_i, a, b, c)

    if g_upper < t_obs or g_lower < t_obs:
        cost_obstacle = 1e6
    else:
        cost_obstacle = -epsilon * (np.log(-g_upper - t_obs) + np.log(-g_lower - t_obs))

    cost_target = gamma_tar_i * np.linalg.norm(z_i - r_i)**2
    cost_agg = gamma_agg_i * np.linalg.norm(z_i - s_i)**2

    return cost_target + cost_agg + cost_obstacle

def cost_gradient(z_i, s_i, r_i, gamma_tar_i, gamma_agg_i, epsilon, a, b, c, t_obs=0.2):
    
    x_i, y_i = z_i
    g_upper = y_i - y_upper(x_i, a, b, c)
    g_lower = - y_i + y_lower(x_i, a, b, c)

    du_dx = 8 * (a * x_i + b)**7 * a
    du_dy = 1
    dl_dx = -8 * (a * x_i + b)**7 * a
    dl_dy = -1

    grad_barrier_upper = np.array([du_dx, du_dy]) / (g_upper - t_obs)
    grad_barrier_lower = np.array([dl_dx, dl_dy]) / (g_lower - t_obs)
    grad_barrier = -epsilon * (grad_barrier_upper + grad_barrier_lower)

    grad_target = 2 * gamma_tar_i * (z_i - r_i)
    grad_aggregate = 2 * gamma_agg_i * (s_i - z_i)

    return grad_target + grad_barrier, grad_aggregate

def phi(z):
    
    return z

class OptimizationNode(Node):
    
    def __init__(self):
        
        super().__init__('optimization_node')

        self.declare_parameter('robot_id', 0)
        self.robot_id = self.get_parameter('robot_id').value
        self.declare_parameter('num_agents', 5)
        self.declare_parameter('alpha', 0.001)
        self.declare_parameter('iterations', 1000)
        self.declare_parameter('random_seed', 42)
        self.declare_parameter('epsilon', 30.0)
        self.declare_parameter('a', 0.3)
        self.declare_parameter('b', -3.0)
        self.declare_parameter('c', 0.5)
        self.declare_parameter('x_max', 5.0)
        self.declare_parameter('x_min', 15.0)
        self.declare_parameter('y_range_robots', [-1.0, 1.0])
        self.declare_parameter('y_range_targets', [-2.0, 2.0])

        #np.random.seed(self.random_seed)
        #self.random_seed = self.get_parameter('random_seed').value

        self.num_agents = self.get_parameter('num_agents').value
        self.G = nx.cycle_graph(self.num_agents)
        self.alpha = self.get_parameter('alpha').value
        self.iterations = self.get_parameter('iterations').value
        self.epsilon = self.get_parameter('epsilon').value
        self.a = self.get_parameter('a').value
        self.b = self.get_parameter('b').value
        self.c = self.get_parameter('c').value
        self.x_max = self.get_parameter('x_max').value
        self.x_min = self.get_parameter('x_min').value
        self.y_range_robots = self.get_parameter('y_range_robots').value
        self.y_range_targets = self.get_parameter('y_range_targets').value
    
        self.robot_positions = generate_robots(self.num_agents, self.x_max, self.y_range_robots)
        self.rr = generate_targets(self.num_agents, self.x_min, self.y_range_targets)
        self.gamma_target = [5 for _ in range(self.num_agents)]
        self.gamma_aggregate = [5 for _ in range(self.num_agents)]
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
            _, gradient_2 = cost_gradient(self.z[i], self.s[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i], self.epsilon, self.a, self.b, self.c)
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

        self.timer = self.create_timer(1.0, self.aggregative_tracking_optimization)

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
            cost = cost_function(self.z[i], self.s[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i], self.epsilon, self.a, self.b, self.c)
            gradient_1, gradient_2 = cost_gradient(self.z[i], self.s[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i], self.epsilon, self.a, self.b, self.c)
            z_new[i] = self.z[i] - alpha * (gradient_1 + np.dot(np.eye(2), self.v[i]))
            s_new[i] = sum(AA[i, j] * self.s[j] for j in range(NN)) + phi(z_new[i]) - phi(self.z[i])
            gradient_1_new, gradient_2_new = cost_gradient(z_new[i], s_new[i], self.rr[i], self.gamma_target[i], self.gamma_aggregate[i], self.epsilon, self.a, self.b, self.c)
            v_new[i] = sum(AA[i, j] * self.v[j] for j in range(NN)) + gradient_2_new - gradient_2
            gradient_2_new_total.append(gradient_2_new)
            gradient_1_total += gradient_1_new
            gradient_2_total += gradient_2_new
            total_cost += cost

        self.z, self.s, self.v = z_new, s_new, v_new
        self.estimates[self.k] = self.s.copy()

        self.costs.append(total_cost)
        
        gradient_total = np.concatenate([gradient_1_total, gradient_2_total])
        self.gradients.append(np.linalg.norm(gradient_total))

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
        msg_positions.data = self.robots[self.k].flatten().tolist()
        self.pub_positions.publish(msg_positions)

        # Publish target positions
        msg_targets = Float64MultiArray()
        msg_targets.data = self.targets[self.k].flatten().tolist()
        self.pub_targets.publish(msg_targets)

        # Publish barycenter estimate position
        barycenter = np.mean(self.s, axis=0)
        msg_barycenter = Float64MultiArray()
        msg_barycenter.data = barycenter.tolist()
        self.pub_barycenter.publish(msg_barycenter)

        # Publish trajectories
        msg_trajectories = Float64MultiArray()
        msg_trajectories.data = self.trajectories[:self.k+1].flatten().tolist()
        self.pub_trajectories.publish(msg_trajectories)

        # # Log info every 100 iterations
        # if self.k % 100 == 0:
        #     self.get_logger().info(f'Iteration {self.k}: Cost = {total_cost}, Gradient norm = {np.linalg.norm(gradient_total)}')
        #     self.get_logger().info(f'Published positions: {msg_positions.data}')
        #     self.get_logger().info(f'Published targets: {msg_targets.data}')
        #     self.get_logger().info(f'Published barycenter: {msg_barycenter.data}')
        #     self.get_logger().info(f'Published trajectories: {msg_trajectories.data}')

        self.k += 1


def main(args=None):
    rclpy.init(args=args)
    node = OptimizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()