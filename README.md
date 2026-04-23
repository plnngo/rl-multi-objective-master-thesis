Run the main.py file to run the tracking and searching objective individually. Therefore, you have to switch the parameter mode to either "track" or search. 
A grid field world will open that contains known (blue) and unknown (yellow) targets. These targets move in accordance to two dynamical patterns, either a linear motion from 
left to right with constant velocity or a constant turn (counter-clockwise) with constant turn rate. The sensor pointing location in the searching objective is modeled by a 
red rectangular that consists of the size of one grid field, representing the field of view. In the case of the tracking objective, a red star highlights the target whose state 
estimate got updated.

The reward term of the searching objective is defined as a constant value of 10 granted for every positive detection. The reward term of the tracking objective is defined as 
the negative sum of all targets' covariance traces. Further information on the modelling of the MDP is to be found in multi_target_env.py.
