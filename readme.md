boosted.rb - AdaBoost.M2 Digit Recognizer 
=========================================

This implementation of AdaBoost.M2 from [Experiments with a New Boosting Algorithm](http://www.cs.princeton.edu/~schapire/publist.html) by Yoav
Freund and Robert E. Schapire is applied to the simulated 7-segment digit
recognition problem presented in [Analysis of the Performance of AdaBoost.M2 for the 
Simulated Digit-Recognition-Example](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.8158&rep=rep1&type=pdf)
 by Gunther Eibl and Karl Peter Pfeiffer.
====
AdaBoosting is a supervised machine learning algorithm used in classification.
Simple hypotheses, the weak learners, are learned iteratively and combined to create
a final hypothesis, the strong learner. Each simple hypothesis has slightly better odds than chance of determining the
correct classification. These simple hypotheses are combined by
scaling and adding them together to produce the final hypothesis.
A very accurate strong learner can be produced by boosting the weak learners.
====
In each boosting iteration, a decision stump is obtained as the simple hypothesis based on a different subset of the given examples.
Each subset is selected from the examples based on a probability distribution over the input that changes in each
boosting iteration. This probability distribution is changed to make it more likely to choose hard-to-classify examples
for the subset.
====
Run from command line
---------------------
ruby boosted.rb [-t number_of_training_examples] [-w number_of_weak_learners] [-b number_of_boosting_iterations]