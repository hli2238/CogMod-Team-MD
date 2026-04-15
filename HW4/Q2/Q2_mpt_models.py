from cmdstanpy import CmdStanModel

model_1ht = CmdStanModel(stan_file="mpt_1ht.stan")
model_2ht = CmdStanModel(stan_file="mpt_2ht.stan")

data = {
    "n_old": 50,
    "n_new": 50,
    "hits": 26,
    "false_alarms": 13
}

fit_1ht = model_1ht.sample(
    data=data,
    chains=4,
    iter_sampling=1000,
    iter_warmup=1000
)

fit_2ht = model_2ht.sample(
    data=data,
    chains=4,
    iter_sampling=1000,
    iter_warmup=1000
)

print(fit_1ht.summary())
print(fit_2ht.summary())