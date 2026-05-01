# Oxford-IIIT-Pet-Classifier - University of York machine learning coursework
### The goal
Given the Oxford-IIT-Pet-Classification dataset, create a from scratch convolution network with the highest testing accuracy in 30 epoches, no external data.

### Hardware/ methodology
I have access to 16 1080ti in 2 servers, therefore I will run different training loops, see the `training_loops directory`, I will document what I have tried, in the `/train_loops/stuff_i_tried.txt`
Multiple training loops is ran concurrently accorss the GPUs on the 2 servers to get the highest testing accuracy on the testing dataset.  
Each training loop will have slightly different parameters/ CNN architecture.

### Current best results
My current best result CNN will be the one in the `gpu.ipynb` with the model saved at `pet_classifer_weights.pth`

