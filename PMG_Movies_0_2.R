##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

#if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
#if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
library(Matrix)
library(compiler)
library(dplyr)
library(stringr)
library(lattice)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                          genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# partition edx
set.seed(1748, sample.kind="Rounding")
test_index = createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-test_index,]
edx_test <- edx[test_index,]

edx_test <- edx_test %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Exploratory Data Analysis - Homework questions

# How many movies?
n_distinct(edx$movieId)
# How many users?
n_distinct(edx$userId)

# How many movies of common genres
sum(str_detect(edx$genres, "Drama"))
sum(str_detect(edx$genres, "Comedy"))
sum(str_detect(edx$genres, "Thriller"))
sum(str_detect(edx$genres, "Romance"))

# The most rated movies
modern <- edx %>% group_by(movieId, title) %>% summarize(count = n()) %>% 
  arrange(desc(count))
quantile(modern$count)

# Ratings by user
user_count <- edx %>% group_by(userId) %>% summarize(count = n()) %>% 
  arrange(desc(count))
quantile(user_count$count)
mean(user_count$count)

# Table and histogram of movie ratings 
rate_count <- edx %>% group_by(rating) %>% summarize(count = n()) %>% 
    arrange(desc(count))
rate_count
rate_count <- rate_count %>% arrange(rating) 
edx %>% ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.5, fill = "blue", col = "black") +
  xlab("Movie Rating")


# Try zeroth model -- just the average rating
tr_avg <- mean(edx_train$rating)

naive <- RMSE(edx_test$rating, tr_avg)
naive

# Add movie effect

movie_avgs <- edx_train %>% group_by(movieId) %>% summarize(b_i = mean(rating - tr_avg))
predicted_ratings <- tr_avg + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
movie <- RMSE(predicted_ratings, edx_test$rating)
movie
movie_avgs %>% ggplot(aes(b_i)) +
  geom_histogram(binwidth = 0.2, fill = "blue", col = "black") +
  xlab("Movie Effect")
quantile(movie_avgs$b_i)

# Add user effect

user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - tr_avg - b_i))

mov_us_predicted_ratings <- edx_test %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = tr_avg +b_i +b_u) %>%
  pull(pred)
mov_us <- RMSE(mov_us_predicted_ratings, edx_test$rating)
mov_us

user_avgs %>% ggplot(aes(b_u)) +
  geom_histogram(binwidth = 0.2, fill = "blue", col = "black") +
  xlab("User Effect")
quantile(user_avgs$b_u)
median(user_avgs$b_u)

# Regularization


# Optimize lambda for user + movie regularization model

lambdas <- seq(2, 7, 0.25)

rmses <- sapply(lambdas, function(l){
  
  tr_avg <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - tr_avg)/(n()+l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - tr_avg)/(n()+l))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = tr_avg + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx_test$rating))
})


lambda_opt <- lambdas[which.min(rmses)]
lambda_opt
qplot(lambdas, rmses)  
best_rmse <- min(rmses)
best_rmse



# Investigate advanced algorithms - matrix factorization etc 
# Matrix algebra and Matrix factorization libraries
library(recosystem)
library(Matrix)

# select user, movie and rating columns from edx and partitions
edx_trips <- edx %>% select(userId, movieId, rating)
edxtr_trips <- edx_train %>% select(userId, movieId, rating)
edxtst_trips <- edx_test %>% select(userId, movieId) %>% mutate(pred = 0)

# Create Recommender System Object

r <- Reco()

# assemble recosystem data files for edx_train and edx_test partition
reco_data <- data_memory(edxtr_trips$userId, edxtr_trips$movieId, edxtr_trips$rating, index1 = TRUE)
reco_test <- data_memory(edxtst_trips$userId, edxtst_trips$movieId, edxtst_trips$rating, index1 = TRUE)

# tune parameters for matrix factorization algorithm

set.seed(1916, sample.kind = "Rounding")

# recosystem tuning function
res =r$tune(reco_data, opt = list(dim =35L, 
                                  costp_l1 = 0,
                                  costp_l2 = c(0.05,0.06, 0.07),
                                  costq_l1 = 0,
                                  costq_l2 = c(0.05,0.06,0.07),
                                  lrate    = 0.1,
                                  nthread = 8,
                                  niter = 40,
                                  nbin = 32,
                                  verbose  = FALSE)
)
res$min

# recosystem training function for matrix factorization model
r$train(reco_data, opts = c(res$min, nthread = 8, nbin = 32, niter = 40, verbose = FALSE))
result =  r$output(out_memory(),out_memory())

# recosystem prediction of internal test partition of edx set
pred_test = r$predict(reco_test, out_memory())
edxtr_MF_error <- RMSE(pred_test, edx_test$rating)
edxtr_MF_error

# Function to calculate user and movie effects for regularized user-model 
reg_user_item <- function(data, l) {
  
  avg <- mean(data$rating)
  
  b_i <- data %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - avg)/(n()+l))
  
  b_u <- data %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - avg)/(n()+l))
  
  predicted_ratings <- 
    data %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = avg + b_i + b_u)
}

# Extract training data user and movie effects
edxtr_pred <- reg_user_item(edx_train, lambda_opt)
edx_pred <- reg_user_item(edx, lambda_opt)
# Form b_u sparse vectors
edxtr_pred_u <- edxtr_pred %>% select(userId, b_u)%>% distinct(userId, .keep_all = TRUE)
edxtr_pred_i <- edxtr_pred %>% select(movieId, b_i) %>% distinct(movieId, .keep_all = TRUE)

# form regularized base model predictions for edx_test
edx_test_trip <- edx_test %>% select(userId, movieId, rating)
edx_test_pred <- 
  edx_test_trip %>%
  left_join(edxtr_pred_i, by = "movieId") %>%
  left_join(edxtr_pred_u, by = "userId")
edx_test_pred <- edx_test_pred %>% mutate(pred = tr_avg + b_u + b_i)
# test weighted average of models

weights <- seq(0, 1, 0.05)

blend_rmses <- sapply(weights, function(w) {
  blend_pred <- w*pred_test + (1-w)*edx_test_pred$pred
  RMSE(blend_pred, edx_test$rating)
})
qplot(weights, blend_rmses)
min(blend_rmses)


# Matrix Factorization of Residuals of Regularized User+Movie model

# Calculate residuals for edx and edxtra
edxtr_res <- edxtr_pred %>% select(userId, movieId, rating, pred) %>% 
  mutate(resid = rating-pred) %>% select(userId, movieId, resid)
edx_res <- edx_pred %>% select(userId, movieId, rating, pred) %>% 
  mutate(resid = rating-pred) %>% select(userId, movieId, resid)

r_resid <- Reco()

# assemble recosystem data matrix for residuals in edx_train partition
reco_resid <- data_memory(edxtr_res$userId, edxtr_res$movieId, edxtr_res$resid, index1 = TRUE)
# tune model for residual matrix
resid_rec =r_resid$tune(reco_resid, opt = list(dim =35L, 
                                               costp_l1 = 0,
                                               costp_l2 = c(0.05,0.06, 0.07),
                                               costq_l1 = 0,
                                               costq_l2 = c(0.05,0.06,0.07),
                                               lrate    = 0.1,
                                               nthread = 8,
                                               niter = 40,
                                               nbin = 32,
                                               verbose  = FALSE)
)
resid_rec$min

# recosystem training function for matrix factorization model

r_resid$train(reco_resid, opts = c(resid_rec$min, nthread = 8, nbin = 32, niter = 40, verbose = FALSE))
result =  r_resid$output(out_memory(),out_memory())

pred_test_res = r_resid$predict(reco_test, out_memory())
edx_test_pred_res <- edx_test_pred %>% mutate(pred_res = pred + pred_test_res)
test_pred_res_error <- RMSE(edx_test_pred_res$pred_res, edx_test$rating)
test_pred_res_error

# select user, movie and rating columns from edx and partitions
edx_trips <- edx %>% select(userId, movieId, rating)

# Create Recommender System Object

r_edx <- Reco()

# assemble recosystem data matrix for full edx dataset
reco_data_edx <- data_memory(edx_trips$userId, edx_trips$movieId, edx_trips$rating, index1 = TRUE)

# tune parameters for matrix factorization algorithm

set.seed(1916, sample.kind = "Rounding")

# recosystem tuning function

res_edx =r_edx$tune(reco_data_edx, opt = list(dim =35L, 
                                              costp_l1 = 0,
                                              costp_l2 = c(0.01,0.04,0.07,0.1),
                                              costq_l1 = 0.01,
                                              costq_l2 = c(0.01,0.04,0.07,0.1),
                                              lrate    = 0.1,
                                              nthread = 8,
                                              niter = 40,
                                              nbin = 32,
                                              verbose  = FALSE)
)
res_edx$min

# Train matrix factorization model on full edx dataset

r_edx$train(reco_data_edx, opts = c(res_edx$min, nthread = 8, nbin = 32, niter = 40, verbose = FALSE))
result =  r_edx$output(out_memory(),out_memory())

# Predict ratings for validation dataset
valid_trip <- validation %>% select(userId, movieId) %>% mutate(pred = 0)
valid_reco <- data_memory(valid_trip$userId, valid_trip$movieId, valid_trip$rating, index1 = TRUE)
pred_valid = r_edx$predict(valid_reco, out_memory())
valid_error <- RMSE(pred_valid, validation$rating)
valid_error