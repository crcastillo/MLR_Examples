############################### Commments on the script #######################
#******************************************************************************
#****
#****   mlr3 MBO - Model Construction and Testing
#****
#****   Objective: Utilize the 1987 National Indonesia Contraceptive Prevalaence
#****   Survey dataset (UCI Machine Learning Repository) to test functionality
#****   of MLR3 MBO hyperparameter optimization. Create a function that can
#****   handle hyperparametertuning, retraining, model comparison, and model
#****   selection. Rebuild the MBO_Tune and MBO_Calibrate functions to be
#****   compatible with mlr3.
#****
#****   6/24/2021 - Initial Build (Chris Castillo)
#****
#****   Code Change Log:
#****   Chris Castillo - 1/30/2020
#****     -
#****
#****   Notes
#****     -
#******************************************************************************
#******************************************************************************

### Establish library sourcing/load, ready workspace, set parameters ====

#* Clean existing space and load libraries
rm(list = ls())
library(mlr3verse)
library(data.table)
library(lubridate)
library(tidyverse)
library(caret)
library(recipes)
library(rsample)
gc()

#* Set parameters
Random_Seed <- 123  # Set the random seed value
Target <- "Contraceptive_Yes"  # Set the target variable
Iterations <- 10L  # Set the number of iterations for MBO to optimize over
TimeBudget <- 60 * 5  # Establish the number of seconds to bound optimization too
Cores <- parallel::detectCores() - 1

### Import data, adjust data types =====

#* Import the dataset
Data <- read.table(
  file = "https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data"
  , header = FALSE
  , sep = ","
  , col.names = c(
    "WifesAge"
    , "WifesEducation" # Low to High
    , "HusbandsEducation" # Low to High
    , "NumberOfChildren"
    , "WifesReligion"  # Non-Islam/Islam
    , "WifeWorking"  # Yes/No
    , "HusbandOccupation"
    , "StandardOfLivingIndex" # Low to High
    , "MediaExposure" # Good/Not Good
    , "ContraceptiveMethod"  # None/Long-Term/Short-Term
  )
  , colClasses = c(
    "integer"
    , "factor"
    , "factor"
    , "integer"
    , "factor"
    , "factor"
    , "factor"
    , "factor"
    , "factor"
    , "factor"
  )
)

#* Add Contraceptive_Yes feature and remove ContraceptiveMethod feature
Data <- Data %>%
  mutate(
    Contraceptive_Yes = as.factor(case_when(
      ContraceptiveMethod == 1 ~ 0
      , TRUE ~ 1
    ))
  ) %>%
  select(
    - ContraceptiveMethod
  )

### Separate Train/Test & Bake the Data ====

#* Create initial split 70/30 Train/Test
set.seed(Random_Seed)
Data_Split <- rsample::initial_split(data = Data, prop = 0.70)

#* Split into Train/Test dataframes
Data_Train <- droplevels(rsample::training(Data_Split))
Data_Test <- droplevels(rsample::testing(Data_Split))

#* Create character vector of "predictor"s and replace the Target location with "outcome"
Role_Vector <- rep(x = "predictor", ncol(Data_Train))
Role_Vector[match(table = names(Data_Train), x = Target)] <- "outcome"

#* Create recipes object for pre-processing and instantiate all step_"s
Rec_Obj <- recipes::recipe(
  x = Data_Train
  , roles = Role_Vector
) %>%
  step_novel(
    all_nominal()
    , -all_outcomes()
    , new_level = NA
  ) %>%
  step_impute_mode(
    all_nominal()
    , -all_outcomes()
  ) %>%
  step_impute_mean(
    all_numeric()
    , -all_outcomes()
  ) %>%
  step_corr(
    all_numeric()
    , -all_outcomes()
    , threshold = 0.90
    , method = "pearson"
  ) %>%
  step_lincomb(
    all_numeric()
    , -all_outcomes()
  ) %>%
  step_dummy(
    all_predictors()
    , -all_numeric()
  ) %>%
  step_zv(
    all_predictors()
  ) %>%
  step_normalize(
    all_predictors()
  ) %>%
  check_missing(
    all_predictors()
  )

#* Create final training prep object that aggregates the recipes together for baking
Trained_Rec_Obj <- recipes::prep(
  x = Rec_Obj
  , training = Data_Train
)

#* Bake (apply) the recipe to the Train/Test dataframes
Data_Train <- recipes::bake(object = Trained_Rec_Obj, Data_Train)
Data_Test <- recipes::bake(object = Trained_Rec_Obj, Data_Test)

#* Create Classification Task
Train_Task <- mlr3::TaskClassif$new(
  id = "Test_Contraceptive"
  , backend = Data_Train
  , target = Target
  , positive = "1"
)

### Tune rpart mlr3 w/ MBO |  ====

#* Create rpart learner
rpart_lrn <- mlr_learners$get(key = "classif.rpart")

#* Print the parameter set for the learner
print(rpart_lrn$param_set)

#* Change predict type of learner to "prob"
rpart_lrn$predict_type <- "prob"

#* Create rpart parameter set
rpart_ps <- mlr3verse::ps(
  cp = paradox::p_dbl(
    lower = 0.001
    , upper = 1
  )
  , minsplit = paradox::p_int(
    lower = 1
    , upper = 30
  )
  , maxdepth = paradox::p_int(
    lower = 1
    , upper = 30
  )
)

#* Create Auto_Tuner object
rpart_at <- mlr3tuning::auto_tuner(
  learner = rpart_lrn
  , method = "mbo"
  , resampling = mlr3::rsmp(
    .key = "cv"
    , folds = 5
    )
  , measure = mlr3::msr("classif.auc")
  , search_space = rpart_ps
  , terminator = bbotk::TerminatorCombo$new(
    terminators = list(
      trm("evals", n_evals = 1000)
      , trm("run_time", secs = 60 * 5)
    )
  )
)

#* Compare runtimes for sequential vs parallel tuning | sequential faster for now
lgr::get_logger("mlr3")$set_threshold("error")
lgr::get_logger("bbotk")$set_threshold("error")
microbm_tuner <- microbenchmark::microbenchmark(
  "Sequential" = {
    progressr::with_progress(
      seq_at <- rpart_at$train(task = Train_Task)
    )
    }
  ,
  "Parallel" = {
    future::plan(
      strategy = future::multisession()
      , workers = Cores
    )
    progressr::with_progress(
      par_at <- rpart_at$train(task = Train_Task)
    )
  }
  , times = 20
)
lgr::get_logger("mlr3")$set_threshold("info")
lgr::get_logger("bbotk")$set_threshold("info")