// Problem 3: logistic regression — Bernoulli likelihood with logit link.
// Problem 4: posterior predictive probabilities for test rows (generated quantities).

data {
  int<lower=0> N;
  int<lower=1> K;
  matrix[N, K] x;
  array[N] int<lower=0, upper=1> y;
  int<lower=0> M;
  matrix[M, K] x_test;
}

parameters {
  real alpha;
  vector[K] beta;
}

model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2);
  y ~ bernoulli_logit(alpha + x * beta);
}
generated quantities {
  vector[M] p_hat;
  if (M > 0) {
    for (m in 1:M) {
      p_hat[m] = inv_logit(alpha + row(x_test, m) * beta);
    }
  }
}