# Neural-Net-Universal-Function-Apporximator
The purpose of this repository is to explore the possibility of building a horseracing model that predicts race outcomes with significant accuracy. The goal is to learn more about machine learning techniques and model optimization with PyTorch and TensorFlow.

Initial assumptions for bulding the model: Individual finishing times are independent, horse finishes are dependent, (are stable positions independent of horse times?)
creating a model that predicts finishing times will be simpler because of independence.

"Learning to Drive" and "Learning to Drive 2.0" utilizes a DQN and trains the model by approximating q-values (or the most optimal next step based on past attempts) using the Bellman equation. At each iteration, the model can opt to accelerate, brake, or turn the model left or right (4 total actions). 
