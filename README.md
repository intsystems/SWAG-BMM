# Project "Optimization operators as evidence estimators"


## Motivation 
"Classical" evidence lower bound approaches allows researcher to perform a simplified Bayesian inference over quite complex models, like deep learning models. This approach involves MC-like sampling at each optimization iteration. Alternative approach is to consider parameters W as a sample from unknown distribution that changes under action of optimization operator (like SGD) at each optimization step. From the researcher perspective, this approach is useufl because doesn't need to change the optimization at all.

## Algorithms to implement 
[ELBO](https://arxiv.org/pdf/1504.01344)

[ELBO with preconditioned SLGD](https://icml.cc/2011/papers/398_icmlpaper.pdf)

Stochastic Gradient Fisher Scoring from [paper](https://www.jmlr.org/papers/volume18/17-214/17-214.pdf)

Constant SGD as Variational EM from [paper](https://www.jmlr.org/papers/volume18/17-214/17-214.pdf)

## Team members
1. Bylinkin Dmitry (Project wrapping, Final demo, Algorithms)
2. Semenov Andrei (Tests writing, Library writing, Algorithms)
3. Shestakov Alexander (Project planning, Blog post, Algorithms)
4. Solodkin Vladimir (Basic code writing, Documentation writing, Algorithms)
