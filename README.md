# Neural-Net-Playground
Playing around with different NN libraries and exploring their capabilities

"Learning to Drive 3.0" trains a car via a reward system and adjusting NN gradients to minimzie MSE of q values i.e. future action taken to maximize rewards moving forward. Adjusting reward system and providing additional information for the model to make decisions.


<img width="307" alt="Screenshot 2023-12-06 at 1 45 41 AM" src="https://github.com/shepard5/Neural-Net-Playground/assets/108085853/1498a9d6-3092-4dbf-b26f-08a43bf80bf2">
 
Successfully crossed the finish line consistently after ~7 minutes training on Apple M1. 

![Figure_1](https://github.com/shepard5/Learning-to-Drive/assets/108085853/86ed0053-f14e-4c60-87ae-eee230909b19)

Consistent completion after 3000 epochs; reward system as a function of arclength of y = sin(x) and large reward for completion, gamma = .9 LR = .01, epsilon = .998 (12/21/23)
