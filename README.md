The project applies the concepts of a hybrid neural network ODE model to the catalytic cracking process.
The network construction approach is taken from [1], and the experimental data are from [2].

A 5-lumped model was constructed (VGO, gases, gasoline, LCO, Coke). The vacuum gas oil reaction was chosen as second-order, while the other processes were considered first-order.
As a result of model training, it was possible to achieve higher accuracy compared to the author of the dissertation [2], who used a classical kinetic model. In particular, the RMSE for predicting gasoline yield decreased from 4.34% to 2.65%, R^2 increased from 0.617 to 0.915.
