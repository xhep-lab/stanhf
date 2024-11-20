Convert hf model to Stan:

    pipx install stanhf
    stanhf ./examples/example.json

Sample from the model, e.g., using HMC

    examples/example sample num_chains=4 data file=examples/example_data.json init=examples/example_init.json

Inspect results using Stan

    STANPATH=$(python -c "from cmdstanpy import cmdstan_path; print(cmdstan_path())") 
    cd ${STANPATH} && make bin/stansummary
    ${STANPATH}/bin/stansummary $(ls *.csv)

or using arviz in Python

    import arviz as az
    import matplotlib.pyplot as plt

    data = az.from_cmdstan([f"output_{i + 1}.csv" for i in range(4)])
    print(az.summary(data))
    az.plot_density(data)
    plt.show()

or using shinystan in R

    install.packages("shinystan")
    install.packages("cmdstanr", repos = c('https://stan-dev.r-universe.dev', getOption("repos")))

    library("cmdstanr")
    csv <- read_cmdstan_csv(c("output_1.csv", "output_2.csv", "output_3.csv", "output_4.csv"))
    mcmc <- as_mcmc.list(csv)

    library("shinystan")
    sso <- as.shinystan(mcmc)
    launch_shinystan(sso)

or using bayesplot in R

    install.packages("bayesplot")
    install.packages("cmdstanr", repos = c('https://stan-dev.r-universe.dev', getOption("repos")))

    library("cmdstanr")
    data <- read_cmdstan_csv(c("output_1.csv", "output_2.csv", "output_3.csv", "output_4.csv"))
    mcmc <- as_mcmc.list(csv)

    library("bayesplot")
    mcmc_violin(mcmc)
