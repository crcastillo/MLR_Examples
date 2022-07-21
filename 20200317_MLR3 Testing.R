############################### Commments on the script ##############################
#*************************************************************************************
#****
#****   MLR3 - Model Construction and Testing
#****
#****   Objective: Utilize the 1987 National Indonesia Contraceptive Prevalaence Survey 
#****   dataset (UCI Machine Learning Repository) to test functionality of MLR3  
#****   hyperparameter optimization. Create a function that can handle hyperparameter
#****   tuning, retraining, model comparison, and selection.
#****
#****   3/17/2020 - Initial Build (Chris Castillo)
#****
#****   Code Change Log:
#****   *User Name* - m/d/yyyy
#****     - 
#****
#****   Notes
#****     - ParamHelpers appears to create conflicts with paradox therefore MBO cannot
#****       be utilized
#****     - Had some issues with ParamHelpers autoloading via a namespace that appears 
#****       have been resolved, very strange and may have to keep an eye out for this
#****       issue creeping up again... I wonder if this is happening when I save a 
#****       RScript... CONFIRMED... Fuck... I have to stop this somehow...
#****         - Addressed by deleting all extra script that calls mlrMBO functions
#****         - RStudio appears to scan the script, assess function calls, and loads 
#****           up the packages via a namespace wihout the users explicit permission
#*************************************************************************************
#*************************************************************************************

### Establish library sourcing/load, ready the workspace, and set parameters ==================

#* Clean existing space and load libraries
rm(list = ls())
library(mlr3verse)
library(tidyverse)
library(tidymodels)
library(magrittr)
library(paradox)
library(future)

# library(lubridate)  # Good
# library(data.table)  # Good
# library(bit64)  # Good
# library(mlrMBO)  # Problematic as ParamHelpers messes with R6 functionality

# library(cgam) # Good
gc()

#* Set parameters
setwd("C:/Users/Chris Castillo/Data Science/Projects/MLR Examples/MBO/")
Random_Seed <- 123  # Set the random seed value
Target <- 'Contraceptive_Yes'  # Set the target variable
Iterations <- 10L  # Set the number of iterations for MBO to optimize over
TimeBudget <- 60 * 15  # Establish the number of seconds to bound optimization too
Cores <- future::availableCores() - 2
Workspace <- './20211230_mlr3_Testing.RData'


### Import data, adjust data types ==================

#* Import the dataset
Data <- read.table(
  file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data'
  , header = FALSE
  , sep = ','  
  , col.names = c(
    'WifesAge'
    , 'WifesEducation' # Low to High
    , 'HusbandsEducation' # Low to High
    , 'NumberOfChildren'
    , 'WifesReligion'  # Non-Islam/Islam
    , 'WifeWorking'  # Yes/No
    , 'HusbandOccupation'
    , 'StandardOfLivingIndex' # Low to High
    , 'MediaExposure' # Good/Not Good
    , 'ContraceptiveMethod'  # None/Long-Term/Short-Term
  )
  , colClasses = c(
    'integer'
    , 'factor'
    , 'factor'
    , 'integer'
    , 'factor'
    , 'factor'
    , 'factor'
    , 'factor'
    , 'factor'
    , 'factor'
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



### Separate Train/Test & Bake the Data ==================

#* Create initial split 70/30 Train/Test
set.seed(Random_Seed)
Data_Split <- rsample::initial_split(data = Data, prop = 0.70)

#* Split into Train/Test dataframes
Data_Train <- droplevels(training(Data_Split))
Data_Test <- droplevels(testing(Data_Split))

#* Create character vector of 'predictor's and replace the Target location with 'outcome'
Role_Vector <- rep(x = 'predictor', ncol(Data_Train))
Role_Vector[match(table = names(Data_Train), x = Target)] <- 'outcome'


#* Create recipes object for pre-processing and instantiate all step_'s
Rec_Obj <- recipes::recipe(
  x = Data_Train
  , roles = Role_Vector
) %>%
  step_novel(
    all_nominal()
    , new_level = NA
  ) %>%
  step_impute_mode(
    all_nominal()
  ) %>%
  step_impute_mean(
    all_numeric()
  ) %>%
  step_corr(
    all_numeric()
    , threshold = 0.90
    , method = 'pearson'
  ) %>%
  step_lincomb(
    all_numeric()
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
  id = "Train_Cirrhosis"
  , backend = Data_Train
  , target = Target
  , positive = '1'
)

#* Test out autoplot functionality
mlr3viz::autoplot(Train_Task)


### Run rpart mlr3 | SUCCESS ==================

#* Create rpart learner
rpart_lrn <- mlr_learners$get(key = 'classif.rpart')

#* Print the parameter set for the learner
print(rpart_lrn$param_set)

#* Change predict type of learner to 'prob'
rpart_lrn$predict_type <- 'prob'

#* Train the learner
rpart_lrn$train(
  task = Train_Task
)

#* Print the trained model 
print(rpart_lrn$model)

#* Create predictions
rpart_pred <- rpart_lrn$predict(
  task = mlr3::TaskClassif$new(
    id = "Test_Cirrhosis"
    , backend = Data_Test
    , target = Target
    , positive = '1'
  )
)

#* Print the confusion matrix
rpart_pred$confusion

#* Use autoplot on the class prediction
autoplot(rpart_pred)  
autoplot(rpart_pred, type = 'roc')  # AUC plot

#* Print out a few measures of the classifier performance
rpart_pred$score(msr("classif.acc")) %>% print()
rpart_pred$score(msr("classif.auc")) %>% print()

#* Save workspace
save.image(Workspace)

### Run rpart mlr3 w/ Resampling | SUCCESS ==================

#* Find the resampling methods
mlr_resamplings %>% as.data.table() %>% print()

#* Instantiate a resampling object
rpart_resample = rsmp(
  'cv'
  , folds = 5L
)

#* Split the task into train/test indices
rpart_resample$instantiate(task = Train_Task)

#* Start parallelization
future::plan(
  strategy = future::multisession(
    workers = Cores
  )
)

#* Run resampling for rpart
Train_resample = mlr3::resample(
  task = Train_Task
  , learner = rpart_lrn
  , resampling = rpart_resample
  , store_models = FALSE
)

# #* Compare runtimes for sequential vs parallel
# microbm <- microbenchmark::microbenchmark(
#   'Sequential' = {Train_resample <- mlr3::resample(
#     task = Train_Task
#     , learner = rpart_lrn
#     , resampling = rsmp(
#       'repeated_cv'
#       , repeats = 10L
#       , folds = 5L
#     )
#     , store_models = FALSE
#   )}
#   , 'Parallel' = {
#     future::plan(
#       strategy = future::multisession(
#         workers = Cores
#       )
#     );
#     Train_resample = mlr3::resample(
#       task = Train_Task
#       , learner = rpart_lrn
#       , resampling = rsmp(
#         'repeated_cv'
#         , repeats = 10L
#         , folds = 5L
#       )
#       , store_models = FALSE
#     )   
#   }
#   , times = 20
# )

### Run rpart mlr3 w/ Hyperparameter Tuning | SUCCESS ==================

#* Create rpart parameter set
rpart_ps = mlr3verse::ps(
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

#* Instantiate a Tuning Instance
rpart_tune = mlr3tuning::TuningInstanceSingleCrit$new(
  task = Train_Task
  , learner = rpart_lrn
  , resampling = mlr3::rsmp(
    .key = 'cv'
    , folds = 5
    )
  , measure = mlr3::msr("classif.auc")
  , search_space = rpart_ps
  , terminator = bbotk::TerminatorCombo$new(
    terminators = list(
      trm('evals', n_evals = 1000)
      , trm('run_time', secs = 60 * 5)
      # , trm('stagnation', iters = 10L, threshold = 1e-05)
    )
  )
)


#* Instantiate a Tuner, 'gensa' doesn't support ParamInt...
rpart_tuner = mlr3tuning::tnr('random_search')

#* Tune it | SUCCESS
rpart_tuner$optimize(
  inst = rpart_tune
)


#* Compare runtimes for sequential vs parallel tuning | sequential faster for now
lgr::get_logger("mlr3")$set_threshold("error")
lgr::get_logger("bbotk")$set_threshold("error")
microbm_tuner <- microbenchmark::microbenchmark(
  'Sequential' = {
    rpart_tuner$optimize(
      inst = rpart_tune
    )
    }
  ,
  'Parallel' = {
    future::plan(
      strategy = future::multisession(
        workers = Cores
      )
    );
    rpart_tuner$optimize(
      inst = rpart_tune
    )
  }
  , times = 20
)
lgr::get_logger("mlr3")$set_threshold("info")
lgr::get_logger("bbotk")$set_threshold("info")


### Run ranger mlr3 w/ Resampling | SUCCESS ==================

#* Create ranger learner
ranger_lrn <- mlr_learners$get(key = 'classif.ranger')

#* Print the parameter set for the learner
ranger_lrn$param_set %>% print()

#* Change predict type of learner to 'prob'
ranger_lrn$predict_type <- 'prob'

#* Instantiate a resampling object
ranger_resample = rsmp(
  'repeated_cv'
  , repeats = 10L
  , folds = 5L
)

#* Split the task into train/test indices
ranger_resample$instantiate(task = Train_Task)

#* Run resampling for ranger
Train_resample_ranger = mlr3::resample(
  task = Train_Task
  , learner = ranger_lrn
  , resampling = ranger_resample
  , store_models = FALSE
)

#* Display aggregate and individual performance
Train_resample_ranger$aggregate(measures = msr('classif.auc')) %>% print()
Train_resample_ranger$score(measures = msr('classif.auc')) %>% print()


#* Compare runtimes for sequential vs parallel
lgr::get_logger("mlr3")$set_threshold("error")
microbm <- microbenchmark::microbenchmark(
  'Sequential' = {Train_resample_ranger <- mlr3::resample(
    task = Train_Task
    , learner = ranger_lrn
    , resampling = rsmp(
      'repeated_cv'
      , repeats = 10L
      , folds = 5L
    )
    , store_models = FALSE
  )}
  , 'Parallel' = {
    future::plan(
      strategy = future::multisession(
        workers = Cores
      )
    );
    Train_resample_ranger = mlr3::resample(
      task = Train_Task
      , learner = ranger_lrn
      , resampling = rsmp(
        'repeated_cv'
        , repeats = 10L
        , folds = 5L
      )
      , store_models = FALSE
    )
  }
  , times = 20
)
lgr::get_logger("mlr3")$set_threshold("info")

### Run ranger mlr3 w/ Hyperparameter Tuning | SUCCESS  ==================

#* Create ranger parameter set
ranger_ps = mlr3verse::ps(
  num.trees = paradox::p_int(
    lower = 1
    , upper = 10
    , default = 5
    , trafo = function(x) x*100
  )
  , mtry = paradox::p_int(
    lower = 3
    , upper = Train_Task$feature_names %>% length() %>% "/"(3) %>% round()
    , default = Train_Task$feature_names %>% length() %>% sqrt() %>% round()
  )
  , min.node.size = paradox::p_int(
    lower = 1
    , upper = 20
  )
  , sample.fraction = paradox::p_dbl(
    lower = 1
    , upper = 19
    , trafo = function(x) round(x, 0) * 0.05
  )
  , replace = paradox::p_lgl()
)

#* Instantiate a Tuning Instance
ranger_tune = mlr3tuning::TuningInstanceSingleCrit$new(
  task = Train_Task
  , learner = ranger_lrn
  , resampling = mlr3::rsmp(
    .key = 'cv'
    , folds = 5
  )
  , measure = mlr3::msr("classif.auc")
  , search_space = ranger_ps
  , terminator = bbotk::TerminatorCombo$new(
    terminators = list(
      trm('evals', n_evals = 1000)
      , trm('run_time', secs = 60 * 10)
    )
  )
)


#* Instantiate a Tuner, 'gensa' doesn't support ParamInt...
ranger_tuner = mlr3tuning::tnr('random_search')

#* Tune it
lgr::get_logger("mlr3")$set_threshold("error")
lgr::get_logger("bbotk")$set_threshold("error")
progressr::with_progress(
  ranger_tuner$optimize(
    inst = ranger_tune
  )
)
lgr::get_logger("mlr3")$set_threshold("info")
lgr::get_logger("bbotk")$set_threshold("info")


#* Save workspace
save.image(Workspace)

#* Take the optimized hyperparameters and set them for the learner
ranger_lrn$param_set$values <- ranger_tune$result_learner_param_vals

#* Re-train the learner with the optimized hyperparameters
ranger_lrn$train(Train_Task)


#* Compare runtimes for sequential vs parallel tuning | sequential faster for now
# lgr::get_logger("mlr3")$set_threshold("error")
# lgr::get_logger("bbotk")$set_threshold("error")
# microbm_tuner <- microbenchmark::microbenchmark(
#   'Sequential' = {
#     rpart_tuner$optimize(
#       inst = rpart_tune
#     )
#   }
#   ,
#   'Parallel' = {
#     future::plan(
#       strategy = future::multisession(
#         workers = Cores
#       )
#     );
#     rpart_tuner$optimize(
#       inst = rpart_tune
#     )
#   }
#   , times = 20
# )
# lgr::get_logger("mlr3")$set_threshold("info")
# lgr::get_logger("bbotk")$set_threshold("info")


#* Score Data_Test with retrained ranger_lrn
Test_Score <- ranger_lrn$predict_newdata(
  newdata = Data_Test
)

#* Show Test performance
Test_Score$score(msr('classif.auc')) %>% print()

### Try out mlr3pipelines |   ==================

#* Recreate the training data for preprocessing
Data_Training <- Data_Split %>% 
  rsample::training() %>% 
  droplevels() %>%
  mlr3::TaskClassif$new(
    id = "Train_Cirrhosis_2"
    , backend = .
    , target = Target
    , positive = '1'
  )

mlr3pipelines::mlr_pipeops %>% as.data.table()

Train_Task$col_roles

# mlr3pipelines::po('fixfactors') %>>%

TEST <- mlr3pipelines::po(
    'imputemode'
    , param_vals = list(
      affect_columns = selector_type("factor")
    )
  ) %>>%
  mlr3pipelines::po(
    'imputemean'
    , param_vals = list(
      affect_columns = selector_type(c("integer", "numeric"))
    )
  ) %>>%
  mlr3pipelines::po(
    'filter'
    , filter = mlr3filters::flt('find_correlation')
    , param_vals = list(
      affect_columns = selector_type(c("integer", "numeric"))
      , filter.cutoff = 0.8
    )
  )


TEST$train(
  input = Data_Training
)


Rec_Obj <- recipes::recipe(
  x = Data_Train
  , roles = Role_Vector
) %>%
  step_novel(
    all_nominal()
    , new_level = NA
  ) %>%
  step_impute_mode(
    all_nominal()
  ) %>%
  step_impute_mean(
    all_numeric()
  ) %>%
  step_corr(
    all_numeric()
    , threshold = 0.90
    , method = 'pearson'
  ) %>%
  step_lincomb(
    all_numeric()
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