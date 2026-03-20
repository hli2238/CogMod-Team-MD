data {
  int<lower=1> N;
  array[N] real x;
  array[N] real y;
}

parameters {
  real alpha;
  real beta;
  real<lower=1e-12> sigma2;
}

transformed parameters {
  real<lower=0> sigma;
  sigma = sqrt(sigma2);
}

model {
  // Priors from assignment
  sigma2 ~ inv_gamma(1, 1);
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);

  // Likelihood
  y ~ normal(alpha + beta * to_vector(x), sigma);
}

generated quantities {
  array[N] real y_rep;
  for (n in 1:N) {
    y_rep[n] = normal_rng(alpha + beta * x[n], sigma);
  }
}
