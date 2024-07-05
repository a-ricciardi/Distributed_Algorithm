The files for running task 2.1 and task 2.1 are inside the python_implementation folder.
In order to run the Python implementation of Task 2 - Aggregative Optimization for Multi-Robot Systems, we must simply run the corresponding files:

1. Run task2_1.py

2. Run task2_3.py

########################################################################################################
The files for running task 2.2 and task 2.4 are inside the src folder.
In order to run the ROS2 implementation of Task 2 - Aggregative Optimization for Multi-Robot Systems, we must procede step by step:

1. Open the terminal and activate ROS2 by typing the following command:
    . /opt/ros/foxy/setup.bash

2. In the same terminal, travel to the directory where the ROS2 packages are contained:
    cd home/task_2

3. Next, build the packages:
    colcon build -- symlink-install

4. Now source the environment:
    . install/setup.bash

5. Task 2.2 can be finally ran by using the launch file:
    ros2 launch task_2_2 task2_2.launch.py

6. To visualize the plots a new terminal must be opened:
    cd home/task_2

7. After traveling to the directory, the graphs can be plotted exploiting a script responsible for this task:
    python3 src/task_2_2/scripts/plots.py

8. Wanting to run task 2.4 (Moving inside a corridor), after setting up the environment, the procedure is the same:
    ros2 launch task_2_4 task2_4.launch.py

9. Again, wanting to visualize the requested plots, there's an identical script that can be ran in a new terminal:
    python3 src/task_2_4/scripts/plots.py

NB: The way the task 2 nodes are designed is such that they save the evolution of the cost, of the gradient norm and of the consensus in files.txt that get 
    overwritten everytime the tasks are re-launched. Also, keep in mind that if the scripts are launched before the iterations are over, the 
    resulting plots are shown until that specific iteration when the script is ran.