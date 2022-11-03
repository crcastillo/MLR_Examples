#* Load libraries
library(mlr3mbo)
library(bbotk)
library(mlr3learners)
library(future)
library(tidyverse)
set.seed(1)

#* Establish cores for parallelization
cores <- parallel::detectCores() - 2

#* Establish objective function
objfun <- bbotk::ObjectiveRFun$new(
    fun = function(xs) list(y = 100 * (xs$x_2 - xs$x_1 ^ 2) ^ 2 + (1 - xs$x_1) ^ 2)
    , domain = paradox::ps(
        x_1 = paradox::p_dbl(
          lower = -10
          , upper = 10
        )
        , x_2 = paradox::p_dbl(
          lower = -10
          , upper = 10
        )
    )
    , codomain = paradox::ps(
        y = paradox::p_dbl(
            tags = "minimize"
        )
    )
)

#* Set the terminator which informs when to end the trial
terminator <- bbotk::trm(
    "evals"
    , n_evals = 1000
)

#* Set the optimization instance
instance <- OptimInstanceSingleCrit$new(
    objective = objfun
    , terminator = terminator
)

#* Define the surrogate learner
surrogate <- SurrogateLearner$new(
  lrn(
    "regr.km"
    , control = list(trace = FALSE)
  )
)

#* Instantiate the acquisition function
acqfun <- AcqFunctionEI$new()

#* Define the acquisition optimizer and terminators
acqopt <- AcqOptimizer$new(
  opt("random_search", batch_size = 100),
  terminator = trm("evals", n_evals = 100)
)

#* Construct the overall optimizer
optimizer <- opt("mbo",
  loop_function = bayesopt_ego,
  surrogate = surrogate,
  acq_function = acqfun,
  acq_optimizer = acqopt
)

#* Run the optimization
optimizer$optimize(instance)


#* Compare runtimes for sequential vs parallel
microbm <- microbenchmark::microbenchmark(
  "Sequential" = {
    sequential_optimizer <- opt("mbo",
        loop_function = bayesopt_ego,
        surrogate = surrogate,
        acq_function = acqfun,
        acq_optimizer = acqopt
        )

    lgr::get_logger("bbotk")$set_threshold("error")
    sequential_optimizer$optimize(instance)
    lgr::get_logger("bbotk")$set_threshold("info")
  }
  , "Parallel" = {
    parallel_optimizer <- opt("mbo",
        loop_function = bayesopt_ego,
        surrogate = surrogate,
        acq_function = acqfun,
        acq_optimizer = acqopt
        )

    lgr::get_logger("bbotk")$set_threshold("error")
    #* Start parallelization
    future::plan(
    strategy = future::multisession()
    , workers = cores
    )
    parallel_optimizer$optimize(instance)
    lgr::get_logger("bbotk")$set_threshold("info")
  }
  , times = 10
)

#* Display the benchmark results
print(microbm)