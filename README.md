# Resource Allocation Reinforcement Learning

![tutorial gif](allocate.ai.gif)

## Overview
This project implements a **Reinforcement Learning (RL)** solution for optimizing the allocation of resources across schools or institutions in a grid-based environment. The goal is to maximize resource distribution efficiency while minimizing unused resources. The agent optimizes the allocation by redistributing resources based on need, simulating a real-world scenario where resource amounts can be unpredictable.

The project uses the **Deep Q-Network (DQN)** algorithm, which allows an agent to learn optimal actions through interactions with the environment. This version focuses on distributing resources across a grid and testing the agent’s ability to balance resources in a set number of iterations.

### Project Structure

- `resource_allocation_env.py`: Contains the custom RL environment where the agent interacts to optimize resource allocation.
- `train.py`: Script to train the DQN agent and save the trained model weights.
- `test_env.py`: A script to test the environment by running episodes with random actions and displaying the state after each action.
- `play.py`: A script that allows the user to input the number of available resources and play with the trained model.

### Installation

To get started, ensure that you have Python 3.9+ installed along with the necessary dependencies.

2.  Set up a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate  
```
Install required libraries:
```
pip install -r requirements.txt   
```
### Dependencies:
- TensorFlow 2.10
- Keras 2.10
- Gym 2.6
- numpy 1.24.3
- Keras-rl2

### Training the Model

To train the agent, run the following command:

```
python train.py
```

The model will train for a specified number of steps (50000 in this case) and save the weights to a file (resource_allocation_dqn_weights.h5). You can modify this number based on your computational power and experiment with training duration.

### Testing the Environment

You can test the environment and agent by running test_env.py, which simulates random actions in the environment for a set number of episodes.

Run the script:
```
python test_env.py
```

The agent will take random actions and render the environment state after each action. You can choose between a terminal or graphical display to view the state.

### Simulating

If you want to interact with the trained model and see how it allocates resources based on user input, you can run the play.py script.

Run the script:
```
python play.py
```
The script will ask you for the number of resources available and simulate the allocation process with the trained agent.

### Known Issues and Limitations

- Handling Negative Numbers: The current version of the agent cannot properly handle negative numbers. in a real world scenario, a school may miss out on a certain distribution of resources. the next batch should propritize that school. 

### Future Improvements

- Handling Negative Numbers: Implement a solution for handling negative resource values, ensuring that the agent can correctly manage cases where resources might become negative.

### Contributing

If you'd like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request. Contributions are welcome!  

### References
[Building a Custom Environment for Deep Reinforcement Learning with OpenAI Gym and Python, Nicholas Renotte](https://www.youtube.com/watch?v=bD6V3rcr_54)


## click here for the [Video Demo](https://drive.google.com/file/d/1cBRCqyjkBPciuOdHeylXSRNwlHkYasKZ/view?usp=sharing)

**Authors:**
* Avit Brian Mugisha

**Affiliation:**
* ALU
