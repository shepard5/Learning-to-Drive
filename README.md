# Neural-Net-Playground
Just playing around with different NN libraries and exploring their capabilities

Horse Racing... Initial assumptions for bulding the model: Individual finishing times are independent, horse finishes are dependent, (are stable positions independent of horse times?)
creating a model that predicts finishing times will be simpler because of independence.

"Learning to Drive" and "Learning to Drive 2.0" is a DQN - trains the model approximating q-values (most optimal next step based on prior info) using the Bellman equation. At each step, car can brake, turn left, right and accel (4 actions). 

Image classification playground templates a convolution NN for any image related datasets - (image formatting in script). 91% cancer identification success rate using pre-optimized hyperparameters from https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset
