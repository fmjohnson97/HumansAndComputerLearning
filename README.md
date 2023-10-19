# HumansAndComputerLearning
code for the class Humans and Computer Learning

To run this code, first download the following repos in the same folder:
- Memory Gym: https://github.com/MarcoMeter/endless-memory-gym
- Memory Maze: https://github.com/jurgisp/memory-maze
- Helm: https://github.com/ml-jku/helm/tree/main

Then, rename their base folders so that the - are _ (Ex: memory-maze would become memory_maze).

Next, clone this repo in the same folder as the previous three. Folders with the same name between this repo and the others should be merged. 

Finally, just run the trainMazeLSTM.py and trainMemGymLSTM.py scripts to train the models, or provide weights (--weights path #path_to_model_weights#) to test a model checkpoint.
