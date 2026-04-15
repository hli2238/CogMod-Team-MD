data {
  int<lower=0> n_old;      
  int<lower=0> n_new;      
  int<lower=0> hits;       
  int<lower=0> false_alarms; 
}

parameters {
  real<lower=0, upper=1> D;  
  real<lower=0, upper=1> g;  
}

transformed parameters {
  real p_hit;
  real p_fa;

  p_hit = D + (1 - D) * g;
  p_fa  = g;
}

model {
  D ~ beta(1,1);
  g ~ beta(1,1);

  hits ~ binomial(n_old, p_hit);
  false_alarms ~ binomial(n_new, p_fa);
}