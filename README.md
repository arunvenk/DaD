# DaD - Data as Demonstrator

[Data as Demonstrator (DaD)](https://www.ri.cmu.edu/pub_files/2015/1/Venkatraman.pdf) is a meta learning algorithm to improve the multi-step predictive capabilities of a learned time series (e.g. dynamical system) model. This method:
- Is simple, easy to implement and can wrap an existing time-series learning procedure.
- Makes no assumption on differentiability. This allows the algorithm to be utilized on top of a larger array of supervised learning algorithms (e.g. random forests & decision trees).
- Is data-efficient in improving a learned model. Without querying the actual system for additional training data, the method is able to achieve better performance on the multi-step criterion by reusing training data to correct for prediction mistakes.
- Can be shown to have performance garauntees that relate the one-step predictive error to the multi-step error.

This repository contains the implementation of the International Symposium on Experimental Robotics ([ISER](http://www.iser2016.org)) 2016 paper: 

> [_Improved Learning of Dynamics for Control_](http://www.cs.cmu.edu/~arunvenk/papers/2016/Venkatraman_iser_16.pdf).
> Arun Venkatraman, Roberto Capobianco, Lerrel Pinto, Martial Hebert, Daniele Nardi, and J. Andrew Bagnell.
> ISER 2016.

DaD was originally presented at [AAAI](http://www.aaai.org/Conferences/AAAI/aaai15.php) 2015: 
> [_Improving multi-step prediction of learned time series models_](https://www.ri.cmu.edu/pub_files/2015/1/Venkatraman.pdf).
> Arun Venkatraman, Martial Hebert, J. Andrew Bagnell. 
> AAAI 2015.

### Main Package:
The main code can be found in the `DaD` folder. Currently, the primary file is `DaD/dad_control.py`. It contains a class `DadControl` which can be used to learn a multi-step predictive model for a controlled dynamical system. Some notes in using this:
- `DaDControl` requires a `learner` object that has a `.fit` and `.predict` method that can take both states and controls. We provide wrappers for using sklearn learners in `DaD.helpers.learner_wrapper`. The demo code has an example.
- The states and controls are passed in as `numpy` tensors with dimensions `[timesteps x dim x num_trajectories]`.
- Passing in `Xtest` and `Utest`is recommended and should be used as a validation dataset. Since there is no monotonic improvement garauntee with DaD, the algorithm tracks the best performance on `Xtest` and returns that model. See the demo code for an example on how to split the data.

> **NOTE:** To use `DaDControl` for time-series problems without controls, one could pass a tensor of zeros for `Utrain` & `Utest`. The code should also be easily modified to remove the controls arguments.

### Demo Code:
A simple demo is provided in the `demos` folder. The demo tries to learn the dynamics of a cartpole being controlled by a randomly generated linear control policy. It can be run by calling 
```ShellSession
python demos/learn_control_demo.py
```
Example results:
```ShellSession
DaD (iters:25). Initial Err: 3.727, Best: 3.211
Err without DaD: 3.175, Err with DaD: 2.484
```
where the `Err` shown corresponds to the RMS error for the multi-step prediction.

#### Using a different learner.
Using a more powerful learner can get us better results. 
```python
from sklearn.neural_network import MLPRegressor
learner = learner_wrapper.DynamicsControlDeltaWrapper(MLPRegressor(hidden_layer_sizes=(20, 10), 
                                        activation='tanh', alpha=1e-3, max_iter=int(1e4), warm_start=False))
```
Using a two-layer network like this, can give us results:
```ShellSession
 DaD (iters:25). Initial Err: 4.096, Best: 2.069
 Err without DaD: 5.534, Err with DaD: 1.983
```
As we can see, the error is a bit lower. Possibly increasing the model complexity or the number of iterations can improve this result.

> **NOTE:** As of August 2016, this requires the latest `sklearn` to get access to the [MLPRegressor](http://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPRegressor.html). This can be installed using
`pip install git+https://github.com/scikit-learn/scikit-learn.git`, though it is recommended to do this in a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) to prevent overwriting the release installation.
