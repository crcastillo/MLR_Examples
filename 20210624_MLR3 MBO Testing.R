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

### Establish library sourcing/load, ready the workspace, and set parameters ==================

#* Clean existing space and load libraries
rm(list = ls())
library(mlr3)

library(ggplot2)
library(lubridate)
library(data.table)
library(bit64)
library(dplyr)
library(caret)
library(rsample)
library(recipes)
library(tibble)
library(magrittr)

library(cgam)
gc()

#* Set parameters
Random_Seed <- 123  # Set the random seed value
Target <- 'Contraceptive_Yes'  # Set the target variable
Iterations <- 10L  # Set the number of iterations for MBO to optimize over
TimeBudget <- 60 * 5  # Establish the number of seconds to bound optimization too
Cores <- parallel::detectCores() - 1
MBO_Folder <- "C:/Users/Chris Castillo/Data Science/Projects/MLR Examples/mlr3 MBO/"


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
    , method = 'pearson'
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
Train_Task <- mlr3::Task$new(
  task_type = 'classif'
  , id = 'Test_Cirrhosis'
  , backend = Data_Train
)

#* Set the target
Train_Task$set_col_roles(
  cols = Target
  , roles = 'target'
)
  

data("mtcars", package = "datasets")
data = mtcars[, 1:3]
str(data)

data %>% class()

task_mtcars = as_task_regr(data, target = "mpg", id = "cars")
print(task_mtcars)

Test <- mlr3::as_task_classif(
  x = Data_Train %>% as.data.frame() %>% class()
  , target = Target
)





### Run ranger MBO | SUCCESS ==================





print(Test)

Test %>%
  print()


?mlr3::Task$new()




#* Change some of the default settings with mlr to ensure tuneParams can contintue to run despite a singular setting error
# mlr::configureMlr(
#   show.info = TRUE
#   , on.learner.error = 'warn'
#   , on.learner.warning = 'quiet'
#   , show.learner.output = FALSE
#   , on.error.dump = FALSE
# )

#* Create learner
ranger_lrn = mlr::makeLearner(
  cl = 'classif.ranger'
  , predict.type = 'prob'
)

#* Define hyperparameters
# ranger_lrn_ps <- ParamHelpers::makeParamSet(
#   makeIntegerParam(
#     id = 'num.trees'
#     , lower = 10
#     , upper = 200
#   )
#   , makeIntegerParam(
#     id = 'min.node.size'
#     , lower = 1
#     , upper = 10
#   )
#   , makeIntegerParam(
#     id = 'mtry'
#     , lower = 1L
#     , upper = floor(sqrt(ncol(Data_Train)))
#   ) 
# )


#* Instantiate MBO control object
# MBO_ctrl <- mlrMBO::makeMBOControl(
#   propose.points = Cores
#   , save.on.disk.at = seq(from = 0, to = Iterations + 1, by = 1)
#   , save.file.path = paste0(
#     MBO_Folder
#     , gsub(pattern = '-', replacement = '', x = Sys.Date())
#     , '_'
#     , format(x = Sys.time(), format = '%H%M%S')
#     , '_TEST_mbo_run.RData'
#   )
# )

#* Establish MBO control termination parameters
# MBO_ctrl <- mlrMBO::setMBOControlTermination(
#   control = MBO_ctrl
#   , iters = Iterations
#   , time.budget = TimeBudget
#   # , max.evals = 14  # Ends optimization if number of evaluations exceed max.evals
# )

#* Establish Infill parameters
# MBO_ctrl <- mlrMBO::setMBOControlInfill(
#   control = MBO_ctrl
#   , crit = crit.cb  # Lower confidence bound, crit.cb2 caused errors
#   # , filter.proposed.points = TRUE  # Ensures proposed points aren't too close to each other, so random hyperparameter points are suggested instead
#   , opt = 'focussearch'
# )


#* Create a mlr Tune Control Object
# MBO_tune_ctrl <- mlr::makeTuneControlMBO(
#   mbo.control = MBO_ctrl
# )

# Store the start time for a parallel run
# start_time_parallel <- Sys.time() 

#* Establish parallelization
# parallelMap::parallelStart(
#   mode = 'socket'
#   , cpus = Cores
# )


#* Attempt using MBO optimization
# tune_MBO <- tuneParams(
#   learner = ranger_lrn
#   , task = Train_Task
#   , resampling = mlr::makeResampleDesc(
#     method = 'CV'
#     , iters = 3
#     , stratify = TRUE
#   )
#   , par.set = ranger_lrn_ps
#   , control = MBO_tune_ctrl
#   , measures = mlr::auc
#   , show.info = TRUE
# )

#* Close parallelization
# parallelMap::parallelStop()

# Store the end time for a parallel run
# end_time_parallel <- Sys.time() 


#* Capture optimization path as data.frame
# tune_MBO_effect <- as.data.frame(tune_MBO$mbo.result$opt.path)

#* Print the time spend on training
# print(end_time_parallel - start_time_parallel)

#* Print the reason for stopping
# print(tune_MBO$mbo.result$final.state)


#* Display example plot of num.trees vs AUC
# ggplot(
#   data = tune_MBO_effect
#   , aes(
#     x = y
#   )) +
#   geom_density() +
#   geom_rug(
#     aes(color = prop.type)
#   ) +
#   ggtitle('Density Plot of AUC by Proposed Points Type') + 
#   xlab('AUC') + 
#   ylab('Density')



### Create catchall mlrMBO function(s) ==================

#* Create the function
# MBO_Tune_Train_Test_Selection = function(
#   learner_list = list()  # List of the learners to be evaluated
#   , hyperparameters_list = list()  # List of the hyperparameters sets to be used with each learne
#   , train_task = NULL  # The MLR Task that will be used for training the models
#   , test_data = NULL  # Define the test data to be used for evaluating tuned models
#   
#   # Set 5-Fold cross-validation with stratified data sets as the resample
#   , resample_method = mlr::makeResampleDesc(  
#     method = 'CV'
#     , iters = 5
#     , stratify = TRUE
#   )
#   , iters = 5L  # Set 5 as the default number of iterations to run
#   , time.budget = NULL  # Define the sum amount of time (in seconds) to budget for the total evaluation
#   , cpus = parallel::detectCores() - 1  # Set prallelization cpus to max cores - 1 as the default
#   , crit = crit.cb  # Set lower confidence bound as the default guide for the MBO optimization process
#   
#   # Set FALSE as the default for using randomly generated hyperparameters when proposed points are too close together
#   # This feature is not supported for hyperparameter sets that have discrete parameters
#   , filter.proposed.points = FALSE  
#   , opt = 'focussearch'  # Set 'focussearch' as the default for the 
#   , measures = list()  # Define the list of comparison measures, only the first element is utilized, the rest are merely evalutated
#   , MBO_save_folder = NULL  # Set the folder location for saving the MBO iteration runs
# ) {
#   
#   #* Change some of the default settings with mlr to ensure tuneParams can contintue to run despite a singular setting error
#   mlr::configureMlr(
#     show.info = TRUE
#     , on.learner.error = 'stop'
#     , on.learner.warning = 'quiet'
#     , show.learner.output = FALSE
#     , on.error.dump = FALSE
#   )  
#   
#   #* Instantiate list() to store objects
#   ReturnList <- learner_list
#   
#   #* Iterate through the list of learners
#   for(Iter in 1:length(ReturnList)){
#     
#     #* Display which learner is being hypertuned
#     print(
#       paste0(
#         'Starting MBO hyperparameter optimization for '
#         , ReturnList[[Iter]]$short.name
#       )
#     )
#     
#     #* Store the hyperparameters sets as an element of learner_list
#     ReturnList[[Iter]]$hyperparameter_set <- hyperparameters_list[[Iter]]
#     
#     #* Store the filepath as an element of learner_list
#     ReturnList[[Iter]]$MBO_save_filepath <- paste0(
#       MBO_save_folder
#       , gsub(pattern = '-', replacement = '', x = Sys.Date())
#       , '_'
#       , format(x = Sys.time(), format = '%H%M%S')
#       , '_'
#       , ReturnList[[Iter]]$short.name
#       , '_mbo_run.RData'
#     )
#     
#     #* Instantiate the MBO control object
#     ReturnList[[Iter]]$MBO_ctrl <- mlrMBO::makeMBOControl(
#       propose.points = cpus
#       , save.on.disk.at = seq(from = 0, to = iters + 1, by = 1)
#       , save.file.path = ReturnList[[Iter]]$MBO_save_filepath
#     )
#     
#     #* Establish the MBO control termination parameters
#     ReturnList[[Iter]]$MBO_ctrl <- mlrMBO::setMBOControlTermination(
#       control = ReturnList[[Iter]]$MBO_ctrl
#       , iters = iters
#       
#       # If cumulative sum of system time elapsed exceeds time.budget then stop after current iteration
#       # Most appropriate when parallelizing the hyperparameter tuning across multiple system cores
#       , time.budget = time.budget
#     )
#     
#     #* Establish Infill parameters for MBO control
#     ReturnList[[Iter]]$MBO_ctrl <- mlrMBO::setMBOControlInfill(
#       control = ReturnList[[Iter]]$MBO_ctrl
#       , crit = crit  # Guides the hyperparameter optimization path 
#       , filter.proposed.points = filter.proposed.points  # Ensures proposed points aren't too close to each other, so random hyperparameter points are suggested instead
#       , opt = opt  # Controls how single points are proposed through the surrogate model
#     )
#     
#     #* Create a Tune Control object
#     ReturnList[[Iter]]$MBO_tune_ctrl <- mlr::makeTuneControlMBO(
#       mbo.control = ReturnList[[Iter]]$MBO_ctrl
#     )
#     
#     
#     
#     #* Store the start time for the parallelized run
#     ReturnList[[Iter]]$parallel_start <- Sys.time()
#     
#     #* Establish parallelization
#     parallelMap::parallelStart(
#       mode = 'socket'
#       , cpus = cpus
#     )
#     
#     #* Run tuneParams
#     ReturnList[[Iter]]$MBO_tune <- mlr::tuneParams(
#       learner = ReturnList[[Iter]]
#       , task = train_task
#       , resampling = resample_method
#       , par.set = ReturnList[[Iter]]$hyperparameter_set
#       , control = ReturnList[[Iter]]$MBO_tune_ctrl
#       , measures = measures
#       , show.info = TRUE
#     )
#     
#     #* Close parallelization
#     parallelMap::parallelStop()
#     
#     #* Store the end time for the parallelized runh
#     ReturnList[[Iter]]$parallel_end <- Sys.time()
#     
#     
#     
#     #* Print the time spent on training
#     print(
#       paste0(
#         'Completed hyperparameter optimization for '
#         , ReturnList[[Iter]]$short.name
#         , ' in '
#         , round(
#           x = difftime(
#             time1 = ReturnList[[Iter]]$parallel_end 
#             , time2 = ReturnList[[Iter]]$parallel_start
#             , units = 'mins'
#           )
#           , digits = 4
#         )
#         , ' minutes due to '
#         , ReturnList[[Iter]]$MBO_tune$mbo.result$final.state
#       )
#     )
#     
#     #* Capture the optimization path as data.frame
#     ReturnList[[Iter]]$MBO_tune_effect <- as.data.frame(ReturnList[[Iter]]$MBO_tune$mbo.result$opt.path)
#     
#     
#     
#     #* Set the par.vals using optimized hyperparameters
#     ReturnList[[Iter]] <- mlr::setHyperPars(
#       learner = ReturnList[[Iter]]
#       , par.vals = ReturnList[[Iter]]$MBO_tune$x
#     )
#     
#     #* Store the start time for the training run
#     ReturnList[[Iter]]$train_start <- Sys.time()
#     
#     #* Retrain the model
#     ReturnList[[Iter]]$trained_model <- mlr::train(
#       learner = ReturnList[[Iter]]
#       , task = train_task
#     )
#     
#     #* Store the end time for the training run
#     ReturnList[[Iter]]$train_end <- Sys.time()
#     
#     #* Print the time spent on retraining
#     print(
#       paste0(
#         'Completed retraining for '
#         , ReturnList[[Iter]]$short.name
#         , ' in '
#         , round(
#           x = difftime(
#             time1 = ReturnList[[Iter]]$train_end 
#             , time2 = ReturnList[[Iter]]$train_start
#             , units = 'mins'
#           )
#           , digits = 4
#         )
#         , ' minutes'
#       )
#     )
#     
#     #* Create predictions using test_data
#     ReturnList[[Iter]]$test_pred <- predict(
#       object = ReturnList[[Iter]]$trained_model
#       , newdata = test_data
#     )
#     
#     #* Evaluate the performance on test_data
#     ReturnList[[Iter]]$test_performance <- mlr::performance(
#       pred = ReturnList[[Iter]]$test_pred
#       , measures = measures
#     )
#     
#   }  # Close loop
#   
#   
#   #* Store the index of the highest performing model
#   if(measures[[1]]$minimize == FALSE){
#     
#     Best_Model <- which.max(
#       unlist(
#         lapply(
#           X = lapply(
#             X = ReturnList
#             , '[['
#             , 'test_performance'
#           )
#           , '[['
#           , 1
#         )
#       )
#     ) 
#   } else {
#     
#     Best_Model <- which.min(
#       unlist(
#         lapply(
#           X = lapply(
#             X = ReturnList
#             , '[['
#             , 'test_performance'
#           )
#           , '[['
#           , 1
#         )
#       )
#     ) 
#     
#   }  # Close boolean
#   
#   
#   #* Print the best model type and performance measure metric
#   print(
#     paste0(
#       'Highest performance model = '
#       , ReturnList[[Best_Model]]$name
#       , ' (Iter = '
#       , Best_Model
#       , ') with '
#       , measures[[1]]$id
#       , ' = '
#       , round(x = ReturnList[[Best_Model]]$test_performance[1], digits = 4)
#     )
#   )
#   
#   #* Return the final object
#   return(ReturnList)
#   
# } # Close function



#* TESTING the function | WORKS
# TEST <- MBO_Tune_Train_Test_Selection(
#   learner_list = list(
#     ada_lrn
#     , ranger_lrn
#     , xgboost_lrn
#     , svm_lrn
#     , glmnet_lrn
#     , cvglmnet_lrn
#   )
#   , hyperparameters_list = list(
#     ada_lrn_ps
#     , ranger_lrn_ps
#     , xgboost_lrn_ps
#     , svm_lrn_ps
#     , glmnet_lrn_ps
#     , cvglmnet_lrn_ps
#   )
#   , train_task = Train_Task
#   , measures = list(
#     mlr::auc
#     , mlr::acc
#   )
#   , cpus = parallel::detectCores() - 2
#   , iters = 5L
#   , time.budget = 60 * 60
#   , test_data = Data_Test
#   , MBO_save_folder = MBO_Folder
# )

#* More in-depth tuning for the selected best using the entire dataset?


### Explore Classifier Calibration | TESTING ==================


# #* Attempt Calibration using predictions on Data_Train
# Train_Pred_Data <- data.frame(
#   y = Train_Task$env$data[[Train_Task$task.desc$target]]
#   , pred = predict(
#     object = TEST[[1]]$trained_model
#     , newdata = Train_Task$env$data
#   )$data$prob.1
# )
# 
# #* Create glm Learner to apply Platt-Scaling (fitting a sigmoid curve through logistic regression)
# Platt_logreg_lrn = mlr::makeLearner(
#   cl = 'classif.logreg'
#   , predict.type = 'prob'
# )
# 
# #* Create Task
# Train_Task_Platt_Scaled <- mlr::makeClassifTask(
#   id = 'Platt_Scaled_logreg'
#   , data = Train_Pred_Data
#   , target = 'y'
# )
# 
# #* Store the trained Platt model
# Platt_Trained_Model <- mlr::train(
#   learner = Platt_logreg_lrn
#   , task = Train_Task_Platt_Scaled
# )
# 
# #* Predict on 
# Platt_Adjusted_Test <- predict(
#   object = Platt_Trained_Model
#   , newdata = data.frame(
#     y = Data_Test[[Target]]
#     , pred = TEST[[1]]$test_pred$data$prob.1)
# )
# 
# 
# #* Show a couple of Calibration plots
# mlr::plotCalibration(
#   obj = mlr::generateCalibrationData(Platt_Adjusted_Test)
#   , smooth = FALSE
# )
# mlr::plotCalibration(
#   obj = mlr::generateCalibrationData(TEST[[1]]$test_pred)
#   , smooth = FALSE
# )
# 
# 
# 
# #* Create evaluation metrics for classifier calibration
# 
# #* Log-Loss, lower is better
# mlr::performance(
#   pred = TEST[[1]]$test_pred
#   , measures = mlr::logloss
# )
# mlr::performance(
#   pred = Platt_Adjusted_Test
#   , measures = mlr::logloss
# )
# 
# #* Brier Score (aka Brier Score Loss), lower is better
# mlr::performance(
#   pred = TEST[[1]]$test_pred
#   , measures = mlr::brier
# )
# mlr::performance(
#   pred = Platt_Adjusted_Test
#   , measures = mlr::brier
# )
# 
# 
# 
# #* Create train data for isotonic regressions
# Isotonic_Train_Data <- data.frame(
#   x = predict(
#     object = TEST[[1]]$trained_model
#     , task = Train_Task
#   )$data$prob.1
#   , y = getTaskData(
#     task = Train_Task
#     , target.extra = TRUE
#   )$target
# )
# 
# 
# #* Use isoreg to perform isotonic regression
# isoreg_fit <- stats::isoreg(
#   x = Isotonic_Train_Data$x
#   , y = Isotonic_Train_Data$y %>%
#     as.character() %>%
#     as.integer()
# )
# 
# #* Use gpava from isotone
# gpava_fit <- isotone::gpava(
#   z = Isotonic_Train_Data$x
#   , y = Isotonic_Train_Data$y %>%
#     as.character() %>%
#     as.integer()
#   , ties = 'primary'
# )
# 
# 
# 
# #* Use cgam from cgam (muhat are fitted values)
# Isotonic_cgam_fitted <- cgam(
#   # formula = y ~ incr.conc(x)  # Increasing with concave splines
#   # formula = y ~ s.incr(x)  # Smooth and increasing
#   formula = y ~ incr(x)  # Increasing monotonic regression with stepchanges (isotonic)
#   # formula = y ~ conc(x)  # Concave splines
#   # formula = y ~ conv(x)  # Convex splines
#   
#   , family = 'binomial'
#   , data = Isotonic_Train_Data
# )
# 
# 
# Isotonic_cgam_fitted$muhat %>% sort() %>% plot()
# 
# 
# 
# Isotonic_cgam_pred <- predict(
#   object = Isotonic_cgam_fitted
#   # , newData = NULL
# )


### Create Custom Learner for Isotonic Regression | TESTING | SUCCESS ==================

# #* Pretty sure I should be using classif instead of regr... 
# #* Confirmed that this should be treated as regression only.
# #* But I need the learner to be classif so I can use embedded mlr measures that are for classif
# 
# #* Create .learner component
# makeRLearner.regr.isotonic = function() {
#   
#   makeRLearnerRegr(
#     cl = 'regr.isotonic'
#     , package = 'stats'
#     , par.set = makeParamSet(
#       # Not sure what to put here since there aren't any parameters to supply outside of y/x (target/feature)
#     )  # Cannot replace with par.set = NULL due to class assertions
#     , properties = c('numerics')
#     , name = 'Isotonic Regression'
#     , short.name = 'isotonic'
#     , note = " There aren't any parameters to adjust "
#   )
#   
# }  # Close function
# 
# #* Create .train component
# trainLearner.regr.isotonic = function(.learner, .task, .subset, .weights = NULL, ...) {
#   
#   d = getTaskData(
#     task = .task
#     , subset = .subset
#     , target.extra = TRUE
#   )
#   
#   stats::isoreg(
#     y = as.matrix(d$target)
#     , x = as.matrix(d$data)
#   )
#   
# }  # Close function
# 
# #* Create .predict component
# predictLearner.regr.isotonic = function(.learner, .model, .newdata, ...) {
#   
#   x = unlist(.newdata)
#   as.stepfun(.model$learner.model)(x)  # No predict method for class = isoreg
#   
# } # Close function


#***********************
#**** TESTING AREA #****
#***********************

# #* Create a Train_Task
# Isotonic_Train_Task <- mlr::makeRegrTask(
#   id = 'Train_Isotonic'
#   , data = data.frame(
#     
#     y = getTaskTargets(task = Train_Task) %>% 
#       as.character() %>% 
#       as.integer()
#     
#     , x = predict(
#       object = TEST[[1]]$trained_model
#       , task = Train_Task
#     )$data$prob.1
#   )
#   , target = 'y'
# )
# 
# #* Create the Learner
# isotonic_lrn <- mlr::makeLearner(
#   cl = 'regr.isotonic'
#   , predict.type = 'response'
# )
# 
# #* Train the Learner
# isotonic_trained <- mlr::train(
#   learner = isotonic_lrn
#   , task = Isotonic_Train_Task
# )
# 
# #* Check the plot of the fitted values
# isotonic_trained$learner.model %>% plot()
# 
# 
# # stats::approxfun(
# #   x = isotonic_trained$learner.model
# #   , method = 'constant'
# #   )(Isotonic_Train_Task$env$data$x) %>% sort() %>% plot()
# # 
# # as.stepfun(isotonic_trained$learner.model)(Isotonic_Train_Task$env$data$x) %>% sort() %>% plot()
# 
# 
# #* Try predict component | SUCCESS | no predict method for class = isoreg
# STORE_NEW <- predict(
#   object = isotonic_trained
#   , newdata = data.frame(
#     x = TEST[[1]]$test_pred$data$prob.1
#   )
# )
# 
# #* Create Test_Task
# Isotonic_Test_Task <- mlr::makeRegrTask(
#   id = 'Test_Isotonic'
#   , data = data.frame(
#     y = as.integer(as.character(Data_Test[[Target]]))
#     , x = TEST[[1]]$test_pred$data$prob.1
#   )
#   , target = 'y'
# )
# 
# #* Predict off Test_Task, works fine!
# STORE_Other <- predict(
#   object = isotonic_trained
#   , task = Isotonic_Test_Task
# )
# 
# 
# 
# ### Work on Classif Build using cgam package | TESTING | SUCCESS ==================
# 
# 
# #* Create .learner component
# makeRLearner.classif.isotonic = function() {
#   
#   makeRLearnerClassif(
#     cl = 'classif.isotonic'
#     , package = 'cgam'
#     , par.set = makeParamSet(
#       makeDiscreteLearnerParam(
#         id = 'shape'
#         , default = 'incr'
#         , values = c(
#           'incr'
#           , 's.incr'
#           , 'conc'
#           , 's.conc'
#           , 'conv'
#           , 's.conv'
#           , 'incr.conc'
#           , 's.incr.conc'
#           , 'incr.conv'
#           , 's.incr.conv'
#         )
#       )
#       , makeDiscreteLearnerParam(
#         id = 'family'
#         , default = 'binomial'
#         , values = c('binomial')
#         , tunable = FALSE
#       )
#       , makeNumericLearnerParam(
#         id = 'cpar'
#         , default = 1.2
#         , lower = 1.0
#         , upper = 2.0
#         , tunable = TRUE
#       )
#       , makeIntegerLearnerParam(
#         id = 'nsim'
#         , default = 100L
#         , lower = 10L
#         , upper = 500L
#         , tunable = TRUE
#       )
#     )
#     , par.vals = list(
#       family = 'binomial'
#       , shape = 'incr'
#     )
#     , properties = c('twoclass', 'numerics', 'factors', 'prob')
#     , name = 'Isotonic Classification'
#     , short.name = 'isotonic'
#     , note = " "
#   )
#   
# }  # Close function
# 
# #* Create .train component
# trainLearner.classif.isotonic = function(.learner, .task, .subset, .weights = NULL, ...) {
#   
#   #* Store the data.frame
#   d = data.frame(
#     getTaskData(
#       task = .task
#       , subset = .subset
#       , target.extra = FALSE
#     )
#   )
#   
#   #* Store formula to be used in model
#   Formula <- as.formula(
#     paste0(
#       mlr::getTaskTargetNames(.task)
#       , ' ~ '
#       , 'cgam::'
#       , .learner$par.vals$shape
#       , '('
#       , mlr::getTaskFeatureNames(.task)
#       , ')'
#     )
#   )
#   
#   #* Run model
#   cgam::cgam(
#     formula = Formula
#     , family = .learner$par.vals$family
#     , data = d
#   )
#   
# }  # Close function
# 
# #* Create .predict component
# predictLearner.classif.isotonic = function(.learner, .model, .newdata, ...) {
#   
#   #* Store the data.frame of .newdata for manipulation
#   df <- data.frame(.newdata)
#   
#   #* Save feature name, max x value, min x value
#   Feature <- .model$features[1]
#   Max_Val <- max(.model$learner.model$xmat0)
#   Min_Val <- min(.model$learner.model$xmat0)
#   
#   #* Handle extreme values as cgam can't extrapolate
#   df[[Feature]][ df[[Feature]] > Max_Val ] <- Max_Val
#   df[[Feature]][ df[[Feature]] < Min_Val ] <- Min_Val
#   
#   #* Make predictions
#   Pred <- cgam::predict.cgam(
#     object = .model$learner.model
#     , newData = df # Works, but I cut off extreme values as cgam can't extrapolate
#   )$fit
#   
#   #* Create matrix with 1 - predictions as sole column
#   Pred_matrix <- matrix(
#     1 - Pred
#     , ncol = 1
#   )
#   
#   #* Column bind matrix with predictions vector
#   Pred_matrix <- cbind(
#     Pred_matrix
#     , matrix(
#       Pred
#       , ncol = 1
#     )
#   )
#   
#   #* Rename the columns after the target factor levels
#   colnames(Pred_matrix) <- .model$factor.levels[[1]]
#   
#   #* Return back the completed matrix
#   return(Pred_matrix)
#   
# } # Close function
# 
# 
# 
# #***********************
# #**** TESTING AREA #****
# #***********************
# 
# #* Create a Train Task
# Isotonic_Train_Task <- mlr::makeClassifTask(
#   id = 'Train_Isotonic'
#   , data = data.frame(
#     y = Data_Train[[Target]]
#     , x = predict(
#       object = TEST[[1]]$trained_model
#       , task = Train_Task
#     )$data$prob.1
#   )
#   , target = 'y'
# )
# 
# #* Create a Test Task
# Isotonic_Test_Task <- mlr::makeClassifTask(
#   id = 'Test_Isotonic'
#   , data = data.frame(
#     y = Data_Test[[Target]]
#     , x = predict(
#       object = TEST[[1]]$trained_model
#       , newdata = Data_Test
#     )$data$prob.1
#   )
#   , target = 'y'
# )
# 
# 
# #* Create the Learner
# isotonic_lrn <- mlr::makeLearner(
#   cl = 'classif.isotonic'
#   , predict.type = 'prob'
#   , shape = 'conc'
# )
# 
# #* Train the Learner
# isotonic_trained <- mlr::train(
#   learner = isotonic_lrn
#   , task = Isotonic_Train_Task
# )
# 
# #* Check the plot of the fitted values
# isotonic_trained$learner.model$muhat %>% sort() %>% plot()
# 
# 
# #* Create predicted values
# pred_values <- predict(
#   object = isotonic_trained
#   # , newdata = Isotonic_Train_Task$env$data # Works
#   # , task = Isotonic_Train_Task # Works
#   , newdata = Isotonic_Test_Task$env$data # Works
#   # , task = Isotonic_Test_Task  # Works
#   # , newdata = data.frame(  # Works
#   #   x = as.numeric(
#   #     c(0.00001, 0.5, 0.9999999)
#   #   )
#   # )
# )
# 
# 
# 
# #* Check the plot of the fitted values
# pred_values$data$prob.1 %>% sort() %>% plot()
# 
# 
# 
# 
# ### Create Function for Classifier Calibration | SUCCESS ==================
# 
# #* Create Calibration Train Task
# Calibrate_Train_Task <- mlr::makeClassifTask(
#   id = 'Calibrate_Train'
#   , data = data.frame(
#     y = Data_Train[[Target]]
#     , x = predict(
#       object = TEST[[1]]$trained_model
#       , task = Train_Task
#     )$data$prob.1
#   )
#   , target = 'y'
# )
# 
# #* Create Calibration Test Task
# Calibrate_Test_Task <- mlr::makeClassifTask(
#   id = 'Calibrate_Test'
#   , data = data.frame(
#     y = Data_Test[[Target]]
#     , x = predict(
#       object = TEST[[1]]$trained_model
#       , newdata = Data_Test
#     )$data$prob.1
#   )
#   , target = 'y'
# )
# 
# #* Set some variables for testing purposes
# # train_task = Calibrate_Train_Task
# # test_task = Calibrate_Test_Task
# # test_pred = TEST[[1]]$test_pred
# # cpus = parallel::detectCores() - 1
# # resample_method = mlr::makeResampleDesc(method = 'RepCV', reps = 5, folds = 5)
# # measures = list(mlr::brier, mlr::logloss)
# 
# 
# #* Create .learner component
# makeRLearner.classif.isotonic = function() {
#   
#   makeRLearnerClassif(
#     cl = 'classif.isotonic'
#     , package = 'cgam'
#     , par.set = makeParamSet(
#       makeDiscreteLearnerParam(
#         id = 'shape'
#         , default = 'incr'
#         , values = c(
#           'incr'
#           , 's.incr'
#           , 'conc'
#           , 's.conc'
#           , 'conv'
#           , 's.conv'
#           , 'incr.conc'
#           , 's.incr.conc'
#           , 'incr.conv'
#           , 's.incr.conv'
#         )
#       )
#       , makeDiscreteLearnerParam(
#         id = 'family'
#         , default = 'binomial'
#         , values = c('binomial')
#         , tunable = FALSE
#       )
#       , makeNumericLearnerParam(
#         id = 'cpar'
#         , default = 1.2
#         , lower = 1.0
#         , upper = 2.0
#         , tunable = TRUE
#       )
#       , makeIntegerLearnerParam(
#         id = 'nsim'
#         , default = 100L
#         , lower = 10L
#         , upper = 500L
#         , tunable = TRUE
#       )
#     )
#     , par.vals = list(
#       family = 'binomial'
#       , shape = 'incr'
#     )
#     , properties = c('twoclass', 'numerics', 'factors', 'prob')
#     , name = 'Isotonic Classification'
#     , short.name = 'isotonic'
#     , note = " "
#   )
#   
# }  # Close function
# 
# #* Create .train component
# trainLearner.classif.isotonic = function(.learner, .task, .subset, .weights = NULL, ...) {
#   
#   #* Store the data.frame
#   d = data.frame(
#     getTaskData(
#       task = .task
#       , subset = .subset
#       , target.extra = FALSE
#     )
#   )
#   
#   #* Store formula to be used in model
#   Formula <- as.formula(
#     paste0(
#       mlr::getTaskTargetNames(.task)
#       , ' ~ '
#       , 'cgam::'
#       , .learner$par.vals$shape
#       , '('
#       , mlr::getTaskFeatureNames(.task)
#       , ')'
#     )
#   )
#   
#   #* Run model
#   cgam::cgam(
#     formula = Formula
#     , family = .learner$par.vals$family
#     , data = d
#   )
#   
# }  # Close function
# 
# #* Create .predict component
# predictLearner.classif.isotonic = function(.learner, .model, .newdata, ...) {
#   
#   #* Store the data.frame of .newdata for manipulation
#   df <- data.frame(.newdata)
#   
#   #* Save feature name, max x value, min x value
#   Feature <- .model$features[1]
#   Max_Val <- max(.model$learner.model$xmat0)
#   Min_Val <- min(.model$learner.model$xmat0)
#   
#   #* Handle extreme values as cgam can't extrapolate
#   df[[Feature]][ df[[Feature]] > Max_Val ] <- Max_Val
#   df[[Feature]][ df[[Feature]] < Min_Val ] <- Min_Val
#   
#   #* Make predictions
#   Pred <- predict.cgam(
#     object = .model$learner.model
#     , newData = df # Works, but I cut off extreme values as cgam can't extrapolate
#   )$fit
#   
#   #* Create matrix with 1 - predictions as sole column
#   Pred_matrix <- matrix(
#     1 - Pred
#     , ncol = 1
#   )
#   
#   #* Column bind matrix with predictions vector
#   Pred_matrix <- cbind(
#     Pred_matrix
#     , matrix(
#       Pred
#       , ncol = 1
#     )
#   )
#   
#   #* Rename the columns after the target factor levels
#   colnames(Pred_matrix) <- .model$factor.levels[[1]]
#   
#   #* Return back the completed matrix
#   return(Pred_matrix)
#   
# } # Close function
# 
# #* Register new learner so it appears under listLearners()
# # registerS3method('makeRLearner', 'classif.isotonic', makeRLearner.classif.isotonic)
# # registerS3method('trainLearner', 'classif.isotonic', trainLearner.classif.isotonic)
# # registerS3method('predictLearner', 'classif.isotonic', predictLearner.classif.isotonic)
# 
# #* Create function
# MLR_Calibrate <- function(
#   train_task = NULL  # The MLR Task that will be used for training the models
#   , test_task = NULL  # Define the MLR Task to be used for evaluating tuned models
#   , test_pred = NULL  # Define the original model predictions on a test set to be evaluated against
#   
#   # Set 5-Fold cross-validation with stratified data sets as the resample
#   , resample_method = mlr::makeResampleDesc(  
#     method = 'RepCV'
#     , reps = 5
#     , folds = 5
#   )
#   , cpus = parallel::detectCores() - 1  # Set prallelization cpus to max cores - 1 as the default
#   
#   # Define the list of comparison measures, only the first element is utilized, the rest are merely evalutated
#   , measures = list(
#     mlr::brier
#     , mlr::logloss
#   )  
# ) { 
#   
#   #* Instantiate list() to store objects
#   ReturnList <- list()
#   
#   
#   #* Create Isotonic Scaling learner
#   isotonic_lrn <- mlr::makeLearner(
#     cl = 'classif.isotonic'
#     , predict.type = 'prob'
#   )
#   
#   #* Define hyperparameters for isotonic learner
#   isotonic_lrn_ps <- ParamHelpers::makeParamSet(
#     makeDiscreteParam(
#       id = 'shape'
#       , values = c(
#         "incr" 
#         , "s.incr"
#         , "conc" 
#         , "s.conc"
#         , "conv" 
#         , "s.conv"
#         , "incr.conc"
#         , "s.incr.conc"
#         , "incr.conv"
#         , "s.incr.conv"
#       )
#     )
#   )
#   
#   
#   # Store the start time for a parallel run
#   # start_time_parallel <- Sys.time()
#   
#   #* Establish parallelization | Can't get to work yet with this custom learner, probably version problems...
#   parallelMap::parallelStart(
#     mode = 'socket'
#     , cpus = cpus
#     , level = 'mlr.tuneParams'
#   )
#   
#   #* Ensure cgam is available on parallelized threads
#   parallelLibrary(
#     packages = 'cgam'
#     , level = 'mlr.tuneParams'
#   )
#   
#   #* Export custom learner to the parallelized threads
#   parallelExport(
#     'trainLearner.classif.isotonic'
#     , 'predictLearner.classif.isotonic'
#     , level = 'mlr.tuneParams'
#   )
#   
#   #* Attempt using MBO optimization | Fails probably due to version differences | Using standard tuneParams for now
#   isotonic_tune_params <- tuneParams(
#     learner = isotonic_lrn
#     , task = train_task
#     , resampling = mlr::makeResampleDesc(
#       method = 'CV'
#       , iters = 5
#       , stratify = TRUE
#     )
#     , par.set = isotonic_lrn_ps
#     , control = mlr::makeTuneControlGrid()
#     , measures = measures
#     , show.info = TRUE
#   )
#   
#   #* Close parallelization
#   parallelMap::parallelStop()
#   
#   #* Store the end time for a parallel run
#   # end_time_parallel <- Sys.time()
#   
#   #* Print the time spend on training
#   # print(end_time_parallel - start_time_parallel)
#   
#   
#   #* Adjust the isotonic learner hyperparameters
#   isotonic_lrn <- mlr::setHyperPars(
#     learner = isotonic_lrn
#     , par.vals = isotonic_tune_params$x
#   )
#   
#   #* Create Platt Scaling learner
#   platt_lrn <- mlr::makeLearner(
#     cl = 'classif.logreg'
#     , predict.type = 'prob'
#   )
#   
#   
#   #* Store the learners in ReturnList
#   ReturnList[[1]] <- isotonic_lrn
#   ReturnList[[2]] <- platt_lrn
#   
#   
#   #* Establish parallelization | Can't get to work yet with this custom learner, probably version problems...
#   parallelMap::parallelStart(
#     mode = 'socket'
#     , cpus = cpus
#     , level = 'mlr.resample'
#   )
#   
#   #* Ensure cgam is available on parallelized threads
#   parallelLibrary(
#     packages = 'cgam'
#     , level = 'mlr.resample'
#   )
#   
#   #* Export custom learner to the parallelized threads
#   parallelExport(
#     'trainLearner.classif.isotonic'
#     , 'predictLearner.classif.isotonic'
#     , level = 'mlr.resample'
#   )
#   
#   #* Derive cross-validated performance for Isotonic Scaling
#   ReturnList[[1]]$cv_perf <- mlr::resample(
#     learner = isotonic_lrn
#     , task = train_task
#     , resampling = resample_method
#     , measures = measures
#     , show.info = TRUE
#   )
#   
#   #* Close parallelization
#   parallelMap::parallelStop()
#   
#   
#   #* Establish parallelization
#   parallelMap::parallelStart(
#     mode = 'socket'
#     , cpus = cpus
#     , level = 'mlr.tuneParams'
#   )
#   
#   #* Derive cross-validated performance for Platt Scaling
#   ReturnList[[2]]$cv_perf <- mlr::resample(
#     learner = platt_lrn
#     , task = train_task
#     , resampling = resample_method
#     , measures = measures
#     , show.info = TRUE
#   )
#   
#   #* Close parallelization
#   parallelMap::parallelStop()
#   
#   
#   #* Store the trained Isotonic model
#   ReturnList[[1]]$trained_model <- mlr::train(
#     learner = isotonic_lrn
#     , task = train_task
#   )
#   
#   #* Store the trained Platt model
#   ReturnList[[2]]$trained_model <- mlr::train(
#     learner = platt_lrn
#     , task = train_task
#   )
#   
#   
#   #* Create predictions using the Isotonic trained model
#   ReturnList[[1]]$test_pred <- predict(
#     object = ReturnList[[1]]$trained_model
#     , task = test_task
#   )
#   
#   #* Create predictions using the Platt trained model
#   ReturnList[[2]]$test_pred <- predict(
#     object = ReturnList[[2]]$trained_model
#     , task = test_task
#   )
#   
#   
#   #* Store the performance for Isotonic
#   ReturnList[[1]]$test_performance <- mlr::performance(
#     pred = ReturnList[[1]]$test_pred
#     , measures = measures
#   )
#   
#   #* Store the performance for Platt
#   ReturnList[[2]]$test_performance <- mlr::performance(
#     pred = ReturnList[[2]]$test_pred
#     , measures = measures
#   )
#   
#   
#   #* Create a vector of the calibration classifier model performance & the original predictions performance
#   Model_Performance <- c(
#     unlist(
#       lapply(
#         X = lapply(
#           X = ReturnList
#           , '[['
#           , base::match(x = 'test_performance', table = names(ReturnList[[1]]))
#         )
#         , '[['
#         , 1
#       )
#     )
#     
#     #* The original prediction performance
#     , mlr::performance(
#       pred = test_pred
#       , measures = measures
#     )[[1]]
#   )
#   names(Model_Performance) <- c('Isotonic', 'Platt', 'Original')
#   
#   
#   #* Identify the best Brier score (lowest) amongst the Isotonic/Platt/Original predictions
#   Best_Model <- which.min(Model_Performance)
#   
#   
#   #* Print the best model type and performance measure metric
#   print(
#     paste0(
#       'Highest performance model = '
#       , names(Model_Performance)[[Best_Model]]
#       , ' with '
#       , measures[[1]]$id
#       , ' = '
#       , round(x = Model_Performance[[Best_Model]], digits = 4)
#     )
#   )
#   
#   # #* Create Calibration Data object for eventual plotting
#   # Calibration_Data <- caret::calibration(
#   #   y ~ x
#   #   , data = test_task$env$data
#   #   , class = 1
#   # )$data %>%
#   #   mutate(
#   #     'Un-Calibrated' = Percent
#   #     , 'Platt' = caret::calibration(
#   #       truth ~ prob.1
#   #       , data = Platt_Test_Pred$data
#   #       , class = 1
#   #       )$data$Percent
#   #     , 'Isotonic' = caret::calibration(
#   #       truth ~ prob.1
#   #       , data = Isotonic_Test_Pred$data
#   #       , class = 1
#   #     )$data$Percent
#   #   ) %>%
#   #   select(-Percent) %>%
#   #   gather_(
#   #     'Stage'
#   #     , 'Percent'
#   #     , c('Un-Calibrated', 'Platt', 'Isotonic')
#   #   )
#   # 
#   # #* Create a calibration plot
#   # ggplot() +
#   #   ggtitle('Calibration Plot') + 
#   #   xlab('Bin Midpoint') +
#   #   geom_line(
#   #     data = Calibration_Data
#   #     , aes(
#   #       x = midpoint
#   #       , y = Percent
#   #       , color = Stage
#   #       )
#   #   ) +
#   #   geom_point(
#   #     data = Calibration_Data
#   #     , aes(
#   #       x = midpoint
#   #       , y = Percent
#   #       , fill = Stage
#   #       , color = Stage
#   #       )
#   #     , size = 3
#   #   ) +
#   #   geom_line(
#   #     aes(
#   #       x = c(0, 100)
#   #       , y = c(0, 100)
#   #     )
#   #     , linetype = 2
#   #     , color = 'grey50'
#   #   )
#   
#   #* Return the final object
#   return(ReturnList)
#   
# }  # Close function
# 
# 
# #* TESTING the function | WORKS
# TEST_Calibrate <- MLR_Calibrate(
#   train_task = Calibrate_Train_Task
#   , test_task = Calibrate_Test_Task
#   , test_pred = TEST[[1]]$test_pred
#   , cpus = parallel::detectCores() - 2
# )
# 
# 
# 
# ### Use CalibratR for Classifier Calibration | TESTING | COMMENTED OUT ==================
# 
# # #* Use CalibratR
# # Test_CalibratR <- CalibratR::calibrate(
# #   actual = STORE$actual
# #   , predicted = STORE$predicted
# #   , folds = 5
# #   , n_seeds = 5
# #   , nCores = parallel::detectCores() - 1
# # )
# # 
# # 
# # names(Test_CalibratR$summary_no_CV$list_errors)
# # 
# # names(Test_CalibratR$summary_no_CV$calibration_error)
# # 
# # names(Test_CalibratR$summary_CV$models$calibrated$BBQ_scaled_avg)
# # 
# # 
# # 
# # CalibratR::statistics_calibratR(Test_CalibratR)
# # 
# # 
# # CalibratR::visualize_calibratR(
# #   calibrate_object = Test_CalibratR
# #   , visualize_models = FALSE
# #   , rd_partitions = TRUE
# # )
# # 
# # names(Test_CalibratR$summary_CV$models$calibrated$GUESS_1)
# # 
# # 
# # CalibratR::compare_models_visual(Test_CalibratR)