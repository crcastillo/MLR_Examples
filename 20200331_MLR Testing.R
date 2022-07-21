############################### Commments on the script ##############################
#*************************************************************************************
#****
#****   MLR MBO - Model Construction and Testing
#****
#****   Objective: Utilize the 1987 National Indonesia Contraceptive Prevalaence Survey 
#****   dataset (UCI Machine Learning Repository) to test functionality of MLR MBO 
#****   hyperparameter optimization. Test out SVM and Neural Network learners.
#****
#****   3/31/2020 - Initial Build (Chris Castillo)
#****
#****   Code Change Log:
#****   *User Name* - m/d/yyyy
#****     - 
#****
#****   Notes
#****     - gaterSVM does not create class probabilities... Worth exploring more down
#****       the road
#****     - neuralnet error if rep > ncol.matrix
#*************************************************************************************
#*************************************************************************************

### Establish library sourcing/load, ready the workspace, and set parameters ==================

#* Clean existing space and load libraries
rm(list = ls())
library(ggplot2)
library(lubridate)
library(data.table)
library(bit64)
library(dplyr)
library(caret)
library(rsample)
library(recipes)
library(tibble)
library(mlrMBO)
library(mlr)
library(parallelMap)
library(parallel)
gc()

#* Set parameters
Random_Seed <- 123  # Set the random seed value
Target <- 'Contraceptive_Yes'  # Set the target variable
Iterations <- 10L  # Set the number of iterations for MBO to optimize over
TimeBudget <- 60 * 60 * 8 # Establish the number of seconds to bound optimization too
Cores <- parallel::detectCores() - 1
MBO_Folder <- "C:/Users/Chris Castillo/Data Science/Projects/MLR Examples/MBO/"


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
  step_modeimpute(
    all_nominal()
    , -all_outcomes()
  ) %>%
  step_meanimpute(
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
Train_Task <- mlr::makeClassifTask(
  id = 'Test_Contraceptive'
  , data = Data_Train
  , target = Target
  , positive = '1'
)


### Configure MLR, construct learner and hyperparameter sets ==================

#* Change some of the default settings with mlr to ensure tuneParams can contintue to run despite a singular setting error
mlr::configureMlr(
  show.info = TRUE
  , on.learner.error = 'stop'
  , on.learner.warning = 'quiet'
  , show.learner.output = FALSE
  , on.error.dump = FALSE
)

#* Create svm learner
svm_lrn = mlr::makeLearner(
  cl = 'classif.svm'
  , predict.type = 'prob'
  # , xval = 5
  # , kernel = 'radial'
)

#* Define svm hyperparameters
svm_lrn_ps <- ParamHelpers::makeParamSet(
  makeNumericParam(
    id = 'cost'
    , lower = -15
    , upper = 15
    , trafo = function(x) 2^x
  )
  , makeNumericParam(
    id = 'gamma'
    , lower = -15
    , upper = 15
    , trafo = function(x) 2^x
  )
  , makeNumericParam(
    id = 'degree'
    , lower = 1
    , upper = 5
    , requires = quote(kernel == 'polynomial')
  )
  , makeDiscreteParam(
    id = 'kernel'
    , values = c('radial', 'linear', 'polynomial')
  )
)


#* Create gaterSVM learner
gaterSVM_lrn = mlr::makeLearner(
  cl = 'classif.gaterSVM'
  , predict.type = 'response'
)


#* Define gaterSVM hyperparameters
gaterSVM_lrn_ps <- ParamHelpers::makeParamSet(
  makeIntegerParam(
    id = 'm' # SVM experts, number of SVM models
    , lower = 2L
    , upper = 5L
  )
  , makeIntegerParam(
    id = 'max.iter'
    , lower = 1L
    , upper = 10L
  )
  , makeIntegerParam(
    id = 'hidden'
    , lower = 5L
    , upper = 20L
  )
  , makeNumericParam(
    id = 'learningrate'
    , lower = 0.001
    , upper = 0.01
  )
)


#* Create evtree learner
evtree_lrn = mlr::makeLearner(
  cl = 'classif.evtree'
  , predict.type = 'prob'
)

#* Define evtree hyperparameters
evtree_lrn_ps <- ParamHelpers::makeParamSet(
  makeIntegerParam(
    id = 'minbucket' # minimum sum of weights in a terminal node
    , lower = 1L
    , upper = 21L
    , default = 7L
  )
  , makeIntegerParam(
    id = 'minsplit'  # minimum sum of weights in a node in order to be considered for splitting
    , lower = 2L
    , upper = 40L
    , default = 20L
  )
  , makeIntegerParam(
    id = 'maxdepth'  # maximum depth of the tree
    , lower = 4L
    , upper = 16L
    , default = 9L
  )
  , makeIntegerParam(
    id = 'ntrees'  # number of trees in population
    , lower = 2L
    , upper = 500L
  )
  , makeNumericParam(
    id = 'alpha'  # regulates the complexity of the cost function
    , lower = 0.1
    , upper = 10
    , default = 1
  )
)


#* Create neuralnet learner
neuralnet_lrn = mlr::makeLearner(
  cl = 'classif.neuralnet'
  , predict.type = 'prob'
  , lifesign = 'minimal'
  , algorithm = 'rprop+'
)

#* Define neuralnet hyperparameters
neuralnet_lrn_ps <- ParamHelpers::makeParamSet(
  # makeIntegerVectorParam(
  #   id = 'hidden' # vector indicating the number of hidden neurons in each layer
  #   , len = 2
  #   , lower = 2
  #   , upper = 4
  #   , trafo = function(x) round(10 ^ (x / 2))
  # )
  makeDiscreteParam(
    id = 'hidden'
    , values = list(
      one_0 = Data_Train %>% ncol()
      , one_1 = Data_Train %>% ncol() %>% sqrt() %>% round()
      , one_2 = Data_Train %>% ncol() %>% '/'(2) %>% round()
      , two_0 = c(
        Data_Train %>% ncol()
        , Data_Train %>% ncol()
        )
      , two_1 = c(
        Data_Train %>% ncol() %>% sqrt() %>% round()
        , Data_Train %>% ncol() %>% sqrt() %>% round()
        )
      , two_2 = c(
        Data_Train %>% ncol() %>% '/'(2) %>% round()
        , Data_Train %>% ncol() %>% '/'(2) %>% round()
        )
      , two_3 = c(
        Data_Train %>% ncol()
        , Data_Train %>% ncol() %>% sqrt() %>% round()
      )
    )
  )
  # , makeIntegerParam(
  #   id = 'rep'  # number of repititions for the network's training
  #   , lower = 1L
  #   , upper = 20L
  #   , default = 1L
  # )
)

### Run SVM MBO | TESTING | SUCCESS ==================

#* Instantiate MBO control object
MBO_ctrl <- mlrMBO::makeMBOControl(
  propose.points = Cores
  , save.on.disk.at = seq(from = 0, to = Iterations + 1, by = 1)
  , save.file.path = paste0(
    MBO_Folder
    , gsub(pattern = '-', replacement = '', x = Sys.Date())
    , '_'
    , format(x = Sys.time(), format = '%H%M%S')
    , '_TEST_mbo_run.RData'
  )
)

#* Establish MBO control termination parameters
MBO_ctrl <- mlrMBO::setMBOControlTermination(
  control = MBO_ctrl
  , iters = Iterations
  , time.budget = TimeBudget
  # , max.evals = 14  # Ends optimization if number of evaluations exceed max.evals
)

#* Establish Infill parameters
MBO_ctrl <- mlrMBO::setMBOControlInfill(
  control = MBO_ctrl
  # , crit = crit.cb  # Lower confidence bound, crit.cb2 caused errors
  # , filter.proposed.points = TRUE  # Ensures proposed points aren't too close to each other, so random hyperparameter points are suggested instead
  , opt = 'focussearch'
)


#* Create a mlr Tune Control Object
MBO_tune_ctrl <- mlr::makeTuneControlMBO(
  mbo.control = MBO_ctrl
)

#* Store the start time for a parallel run
start_time_parallel <- Sys.time()

#* Establish parallelization
parallelMap::parallelStart(
  mode = 'socket'
  , cpus = Cores
)


#* Attempt using MBO optimization
tune_MBO <- tuneParams(
  learner = svm_lrn
  , task = Train_Task
  , resampling = mlr::makeResampleDesc(
    method = 'CV'
    , iters = 3
    , stratify = TRUE
  )
  , par.set = svm_lrn_ps
  , control = MBO_tune_ctrl
  , measures = mlr::auc
  , show.info = TRUE
)

#* Close parallelization
parallelMap::parallelStop()

# Store the end time for a parallel run
end_time_parallel <- Sys.time() 


#* Capture optimization path as data.frame
tune_MBO_effect <- as.data.frame(tune_MBO$mbo.result$opt.path)

#* Print the time spend on training
print(end_time_parallel - start_time_parallel)

#* Print the reason for stopping
print(tune_MBO$mbo.result$final.state)


#* Display example plot of num.trees vs AUC
ggplot(
  data = tune_MBO_effect[ , !duplicated(colnames(tune_MBO_effect))]
  , aes(
    x = y
  )) +
  geom_density() +
  geom_rug(
    aes(color = prop.type)
  ) +
  ggtitle('Density Plot of AUC by Proposed Points Type') + 
  xlab('AUC') + 
  ylab('Density')

### Run gaterSVM MBO | TESTING | SUCCESS, but doesn't create probs ==================

#* Instantiate MBO control object
MBO_ctrl <- mlrMBO::makeMBOControl(
  propose.points = Cores
  , save.on.disk.at = seq(from = 0, to = Iterations + 1, by = 1)
  , save.file.path = paste0(
    MBO_Folder
    , gsub(pattern = '-', replacement = '', x = Sys.Date())
    , '_'
    , format(x = Sys.time(), format = '%H%M%S')
    , '_TEST_mbo_run.RData'
  )
)

#* Establish MBO control termination parameters
MBO_ctrl <- mlrMBO::setMBOControlTermination(
  control = MBO_ctrl
  , iters = Iterations
  , time.budget = TimeBudget
  # , max.evals = 14  # Ends optimization if number of evaluations exceed max.evals
)

#* Establish Infill parameters
MBO_ctrl <- mlrMBO::setMBOControlInfill(
  control = MBO_ctrl
  # , crit = crit.cb  # Lower confidence bound, crit.cb2 caused errors
  # , filter.proposed.points = TRUE  # Ensures proposed points aren't too close to each other, so random hyperparameter points are suggested instead
  , opt = 'focussearch'
)


#* Create a mlr Tune Control Object
MBO_tune_ctrl <- mlr::makeTuneControlMBO(
  mbo.control = MBO_ctrl
)

#* Store the start time for a parallel run
start_time_parallel <- Sys.time()

#* Establish parallelization
parallelMap::parallelStart(
  mode = 'socket'
  , cpus = Cores
)


#* Attempt using MBO optimization
tune_MBO <- tuneParams(
  learner = gaterSVM_lrn
  , task = Train_Task
  , resampling = mlr::makeResampleDesc(
    method = 'CV'
    , iters = 3
    , stratify = TRUE
  )
  , par.set = gaterSVM_lrn_ps
  , control = MBO_tune_ctrl
  , measures = mlr::acc
  , show.info = TRUE
)

#* Close parallelization
parallelMap::parallelStop()

# Store the end time for a parallel run
end_time_parallel <- Sys.time() 


#* Capture optimization path as data.frame
tune_MBO_effect <- as.data.frame(tune_MBO$mbo.result$opt.path)

#* Print the time spend on training
print(end_time_parallel - start_time_parallel)

#* Print the reason for stopping
print(tune_MBO$mbo.result$final.state)


#* Display example plot of num.trees vs AUC
ggplot(
  data = tune_MBO_effect[ , !duplicated(colnames(tune_MBO_effect))]
  , aes(
    x = y
  )) +
  geom_density() +
  geom_rug(
    aes(color = prop.type)
  ) +
  ggtitle('Density Plot of AUC by Proposed Points Type') + 
  xlab('AUC') + 
  ylab('Density')


### Run evtree MBO | TESTING | SUCCESS ==================

#* Instantiate MBO control object
MBO_ctrl <- mlrMBO::makeMBOControl(
  propose.points = Cores
  , save.on.disk.at = seq(from = 0, to = Iterations + 1, by = 1)
  , save.file.path = paste0(
    MBO_Folder
    , gsub(pattern = '-', replacement = '', x = Sys.Date())
    , '_'
    , format(x = Sys.time(), format = '%H%M%S')
    , '_TEST_mbo_run.RData'
  )
)

#* Establish MBO control termination parameters
MBO_ctrl <- mlrMBO::setMBOControlTermination(
  control = MBO_ctrl
  , iters = Iterations
  , time.budget = TimeBudget
  # , max.evals = 14  # Ends optimization if number of evaluations exceed max.evals
)

#* Establish Infill parameters
MBO_ctrl <- mlrMBO::setMBOControlInfill(
  control = MBO_ctrl
  # , crit = crit.cb  # Lower confidence bound, crit.cb2 caused errors
  # , filter.proposed.points = TRUE  # Ensures proposed points aren't too close to each other, so random hyperparameter points are suggested instead
  , opt = 'focussearch'
)


#* Create a mlr Tune Control Object
MBO_tune_ctrl <- mlr::makeTuneControlMBO(
  mbo.control = MBO_ctrl
)

#* Store the start time for a parallel run
start_time_parallel <- Sys.time()

#* Establish parallelization
parallelMap::parallelStart(
  mode = 'socket'
  , cpus = Cores
)


#* Attempt using MBO optimization
tune_MBO <- tuneParams(
  learner = evtree_lrn
  , task = Train_Task
  , resampling = mlr::makeResampleDesc(
    method = 'CV'
    , iters = 3
    , stratify = TRUE
  )
  , par.set = evtree_lrn_ps
  , control = MBO_tune_ctrl
  , measures = mlr::auc
  , show.info = TRUE
)

#* Close parallelization
parallelMap::parallelStop()

# Store the end time for a parallel run
end_time_parallel <- Sys.time() 


#* Capture optimization path as data.frame
tune_MBO_effect <- as.data.frame(tune_MBO$mbo.result$opt.path)

#* Print the time spend on training
print(end_time_parallel - start_time_parallel)

#* Print the reason for stopping
print(tune_MBO$mbo.result$final.state)


#* Display example plot of num.trees vs AUC
ggplot(
  data = tune_MBO_effect[ , !duplicated(colnames(tune_MBO_effect))]
  , aes(
    x = y
  )) +
  geom_density() +
  geom_rug(
    aes(color = prop.type)
  ) +
  ggtitle('Density Plot of AUC by Proposed Points Type') + 
  xlab('AUC') + 
  ylab('Density')

### Run neuralnet MBO | TESTING |  ==================

#* Instantiate MBO control object
MBO_ctrl <- mlrMBO::makeMBOControl(
  propose.points = Cores
  , save.on.disk.at = seq(from = 0, to = Iterations + 1, by = 1)
  , save.file.path = paste0(
    MBO_Folder
    , gsub(pattern = '-', replacement = '', x = Sys.Date())
    , '_'
    , format(x = Sys.time(), format = '%H%M%S')
    , '_TEST_mbo_run.RData'
  )
)

#* Establish MBO control termination parameters
MBO_ctrl <- mlrMBO::setMBOControlTermination(
  control = MBO_ctrl
  , iters = Iterations
  , time.budget = TimeBudget
  # , max.evals = 14  # Ends optimization if number of evaluations exceed max.evals
)

#* Establish Infill parameters
MBO_ctrl <- mlrMBO::setMBOControlInfill(
  control = MBO_ctrl
  # , crit = crit.cb  # Lower confidence bound, crit.cb2 caused errors
  # , filter.proposed.points = TRUE  # Ensures proposed points aren't too close to each other, so random hyperparameter points are suggested instead
  , opt = 'focussearch'
)


#* Create a mlr Tune Control Object
MBO_tune_ctrl <- mlr::makeTuneControlMBO(
  mbo.control = MBO_ctrl
)

#* Store the start time for a parallel run
start_time_parallel <- Sys.time()

#* Establish parallelization
parallelMap::parallelStart(
  mode = 'socket'
  , cpus = Cores
)


#* Attempt using MBO optimization
tune_MBO <- tuneParams(
  learner = neuralnet_lrn
  , task = Train_Task
  , resampling = mlr::makeResampleDesc(
    method = 'CV'
    , iters = 3
    , stratify = TRUE
  )
  , par.set = neuralnet_lrn_ps
  , control = MBO_tune_ctrl
  , measures = mlr::auc
  , show.info = TRUE
)

#* Close parallelization
parallelMap::parallelStop()

# Store the end time for a parallel run
end_time_parallel <- Sys.time() 


#* Capture optimization path as data.frame
tune_MBO_effect <- as.data.frame(tune_MBO$mbo.result$opt.path)

#* Print the time spend on training
print(end_time_parallel - start_time_parallel)

#* Print the reason for stopping
print(tune_MBO$mbo.result$final.state)


#* Display example plot of num.trees vs AUC
ggplot(
  data = tune_MBO_effect[ , !duplicated(colnames(tune_MBO_effect))]
  , aes(
    x = y
  )) +
  geom_density() +
  geom_rug(
    aes(color = prop.type)
  ) +
  ggtitle('Density Plot of AUC by Proposed Points Type') + 
  xlab('AUC') + 
  ylab('Density')