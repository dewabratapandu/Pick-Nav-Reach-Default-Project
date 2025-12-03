# README!!
There are two main files in this repository:
- `pick_nav_reach_env.py` for antipodal grasping
- `pick_gym_env.py` for SAC grasping

# How to test RL-based grasping mechanism and navigration through the grid
1. Download the model file: https://drive.google.com/file/d/1uvOMd-7vwug5uu22OZSwakHCSaUcycvp/view?usp=drive_link
2. Place it in the root directory or otherwise change path in `pick_gym_env.py` -> `model = SAC.load("./best_model_submittable_final.zip", env=env)`
3. Run python pick_gym_env.py

Object Ids that can be grasped: [2, 3, 4, 7, 10]

![Scene](imgs/scene.png)


# Installation

1. Create a conda environment using `Python 3.10`.

```
conda create -n pnr python==3.10
conda activate pnr
```

2. Our environment is build on [PyBullet](https://pybullet.org/wordpress/index.php/forum-2/). Install it with pip:

```
pip3 install pybullet numpy matplotlib trimesh
```

3. Clone the project repo:

```
git clone https://github.com/NUS-LinS-Lab/Pick-Nav-Reach-Default-Project.git
```

# Run the Environment 

`python pick_nav_reach_env.py`

You should replace the `keyboard_controller` with a customized module designed by yourselves and then use `env.step()` to controll the robot to complete the task.

`env.step()` takes in a `numpy.ndarray` of shape (15,) as **action**, where 15 is the number of DoFs of the Fetch robot. The **observation** and **extra information** will be returned in python dictionaries as follows:

```python
# key_name: value_type, value_size

# observation dict
qpos: <class 'numpy.ndarray'>, (15,) # (num_dofs,)
qvel: <class 'numpy.ndarray'>, (15,) # (num_dofs,)
object_pos: <class 'numpy.ndarray'>, (3,) # position
object_xyzw: <class 'numpy.ndarray'>, (4,) # quaternion xyzw
goal_pos: <class 'numpy.ndarray'>, (3,) # position
object_pc: <class 'trimesh.caching.TrackedArray'>, (1024, 3) # (num_points, position)
object_normals: <class 'numpy.ndarray'>, (1024, 3) # (num_points, direction)
cube_positions: <class 'numpy.ndarray'>, (140, 2) # (num_cubes, xy_position)

# info dict
dist_to_goal: <class 'numpy.float64'>, () # < 0.1 means success
success: <class 'numpy.bool'>, () # success or not

```

# Requirement

You should implement the grasp generation, motion planning, and navigation algorithms in your customized module by yourselves to accomplish the task.

# Rubrics

We evaluate your algorithms in the following two aspects with 5 Unprovided seeds:

- Pick: 5 test objects (within the provided objects).

- Navigation and Reach: 5 test mazes.

Unless otherwise stated: placement tolerance = **10 cm**

## Pick (64%)

### Grasp Genration 

Use the **object point cloud and surface normals** to generate grasps.

- Function for Grasp Generation - 20 pts

### Motion Planning and Grasping

Use motion planning to execute the generated grasps.

- Function for Motion Planning - 20 pts

For each trial, award the following (sum; then average across trials):

- Lift within 3000 sim steps — 6 pts

- Lift height ≥ 10 cm above table — 6 pts

- Hold stability for ≥ 50 steps — 6 pts

- Final pose close to target grasp — 6 pts

![Scene](imgs/success_grasp.png)

## Navigation and Reach (36%)

Use the maze configuration to navigate and reach the goal.

- Function for Navigation - 20 pts

For each trial, award the following (sum; then average across trials):

- reach within 10 cm of the goal sphere — 6 pts

- reach within 15000 sim steps - 6 pts

- Carry Stability — 4 pts

    no drop that requires re-grasp during navigation → 2 pts each (max 8).

![Scene](imgs/success_navigation.png)


# References

- [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.2ye70wns7io3).

# Acknowledgments

- YCB object models are chosen and adapted from [here](https://www.ycbbenchmarks.com/).
