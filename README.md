# Oxford-IIIT-Pet-Classifier - University of York machine learning coursework
### For the marker
If you are to check my progress history, please check the  also check the commit history for `train.ipynb` and the `train.py`.
I started development on the `train.ipynb` then moved on to the `train.py`

### The goal
Given the Oxford-IIT-Pet-Classification trainval dataset, create a from scratch convolution network with the highest testing accuracy in 30 epoches, no external data.

### Hardware/ methodology
I have access to 15 1080ti in 2 servers, therefore I will run training loops concurrently, see the `training_loops directory`, I will document what I have tried, in the `/train_loops/stuff_i_tried.txt`
Multiple training loops is ran concurrently across the GPUs on the 2 servers to get the highest testing accuracy on the testing dataset.  
Each training loop will have slightly different hyperparameters/ CNN architecture.  
I prevent myself from using the testing results to edit my CNN, so only uses the validation accuracy to do determine what to change, to prevent the testing dataset leakage.  
My final model at `train.py` is without the valuation split for maximum training data for maximum accuracy, achieving 72.20% testing accuracy.

### Current best results
My current best result CNN will be the one in the `train.ipynb` with the model saved at `pet_classifer_weights.pth`
