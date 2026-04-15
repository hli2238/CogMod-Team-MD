data {
  int<lower=0> n_old;
  int<lower=0> n_new;
  int<lower=0> hits;
  int<lower=0> false_alarms;
}

parameters {
  real<lower=0, upper=1> D1;  
  real<lower=0, upper=1> D2;  
  real<lower=0, upper=1> g;   
}

transformed parameters {
  real p_hit;
  real p_fa;

  p_hit = D1 + (1 - D1) * g;
  p_fa  = (1 - D2) * g;
}

model {
  D1 ~ beta(1,1);
  D2 ~ beta(1,1);
  g  ~ beta(1,1);

  hits ~ binomial(n_old, p_hit);
  false_alarms ~ binomial(n_new, p_fa);
}