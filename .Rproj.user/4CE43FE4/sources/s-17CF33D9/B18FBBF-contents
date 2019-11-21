library(RVerbalExpressions)
library(textclean)
library(tidyverse)
library(tidytext)
library(rsample)
library(caret)

# import data
data <- read_csv("data/tweets.csv")
head(data)


# Text Normalization


mention <- rx() %>% 
  rx_find(value = "@") %>% 
  rx_alnum() %>% 
  rx_one_or_more()
mention


"@VirginAmerica What @dhepburn said." %>% 
  str_remove_all(pattern = mention) %>% 
  str_squish()


#### hashtag

hashtag <- rx() %>% 
  rx_find(value = "#") %>% 
  rx_alnum() %>% 
  rx_one_or_more()
hashtag


"@VirginAmerica I'm #elevategold for a good reason: you rock!!" %>% 
  str_remove_all(pattern = mention) %>%
  str_remove_all(pattern = hashtag) %>% 
  str_squish()

#### question mark

question <- rx() %>% 
  rx_find(value = "?") %>% 
  rx_one_or_more()
question


#### exclamation mark


exclamation <- rx() %>% 
  rx_find(value = "!") %>% 
  rx_one_or_more()
exclamation


#### punctuation


punctuation <- rx_punctuation()
punctuation


#### number


number <- rx_digit()
number


#### dollar sign


dollar <- rx() %>% 
  rx_find("$")
dollar



### Text Cleansing {.tabset}

#### `replace_url`


"@VirginAmerica Really missed a prime opportunity, there. https://t.co/mWpG7grEZP" %>% 
  replace_url()


#### `replace_emoticon`


"@SouthwestAir thanks! Very excited to see it :3" %>%
  replace_emoticon()


#### `replace_contruction`


"@united I'd thank you - but you didn't help. taking 6 hours to reply isn't actually helpful" %>% 
  replace_contraction()


#### `replace_word_elongation`


"@VirginAmerica heyyyy guyyyys.. :/" %>% 
  replace_word_elongation()



data <- data %>% 
  mutate(
    text_clean = text %>% 
      replace_url() %>% 
      replace_emoji() %>% 
      replace_emoticon() %>% 
      replace_html() %>% 
      str_remove_all(pattern = mention) %>% 
      str_remove_all(pattern = hashtag) %>% 
      replace_contraction() %>% 
      replace_word_elongation() %>% 
      str_remove_all(pattern = punctuation) %>% 
      str_remove_all(pattern = number) %>% 
      str_remove_all(pattern = dollar) %>% 
      str_to_lower() %>% 
      str_squish()
  )



data %>% 
  select(text, text_clean) %>% 
  head()


### Term Frequency Plot


top_positive <- data %>% 
  filter(airline_sentiment == "positive") %>% 
  unnest_tokens(word, text_clean) %>% 
  count(word, airline_sentiment, sort = TRUE) %>% 
  filter(!(word %in% stopwords::stopwords(language = "en"))) %>% 
  head(20)

top_negative <- data %>% 
  filter(airline_sentiment == "negative") %>% 
  unnest_tokens(word, text_clean) %>% 
  count(word, airline_sentiment, sort = TRUE) %>% 
  filter(!(word %in% stopwords::stopwords(language = "en"))) %>% 
  head(20) 



bind_rows(top_positive, top_negative) %>% 
  ggplot(mapping = aes(x = reorder(word, n), y = n)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~airline_sentiment, scales = "free") +
  theme_minimal() +
  labs(x = "Terms",
       y = "Freq")


### Wordcloud





## Modeling


# prepare datainput
data <- data %>% 
  filter(airline_sentiment %in% c("positive", "negative")) %>% 
  mutate(label = factor(airline_sentiment, levels = c("negative", "positive")),
         label = as.numeric(label),
         label = label - 1) %>% 
  select(text_clean, label) %>% 
  na.omit()
head(data, 10)



num_words <- 1024 

# prepare tokenizers
tokenizer <- text_tokenizer(num_words = num_words,
                            lower = TRUE) %>% 
  fit_text_tokenizer(data$text_clean)



set.seed(100)
intrain <- initial_split(data = data, prop = 0.8, strata = "label")

data_train <- training(intrain)
data_test <- testing(intrain)

set.seed(100)
inval <- initial_split(data = data_test, prop = 0.5, strata = "label")

data_val <- training(inval)
data_test <- testing(inval)



maxlen <- max(str_count(data$text_clean, "\\w+")) + 1 



# prepare x
data_train_x <- texts_to_sequences(tokenizer, data_train$text_clean) %>%
  pad_sequences(maxlen = maxlen)

data_val_x <- texts_to_sequences(tokenizer, data_val$text_clean) %>%
  pad_sequences(maxlen = maxlen)

data_test_x <- texts_to_sequences(tokenizer, data_test$text_clean) %>%
  pad_sequences(maxlen = maxlen)

# prepare y
data_train_y <- to_categorical(data_train$label, num_classes = 3)
data_val_y <- to_categorical(data_val$label, num_classes = 3)
data_test_y <- to_categorical(data_test$label, num_classes = 3)



# initiate keras model sequence
model <- keras_model_sequential()

# model
model %>%
  # layer input
  layer_embedding(
    name = "input",
    input_dim = num_words,
    input_length = maxlen,
    output_dim = 32, 
    embeddings_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2)
  ) %>%
  # layer dropout
  layer_dropout(
    name = "embedding_dropout",
    rate = 0.5
  ) %>%
  # layer lstm 1
  layer_lstm(
    name = "lstm",
    units = 256,
    dropout = 0.2,
    recurrent_dropout = 0.2,
    return_sequences = FALSE, 
    recurrent_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2),
    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2)
  ) %>%
  # layer output
  layer_dense(
    name = "output",
    units = 3,
    activation = "softmax", 
    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2)
  )



# compile the model
model %>% compile(
  optimizer = "adam",
  metrics = "accuracy",
  loss = "categorical_crossentropy"
)

# model summary
summary(model)



# model fit settings
epochs <- 10
batch_size <- 512

# fit the model
history <- model %>% fit(
  data_train_x, data_train_y,
  batch_size = batch_size, 
  epochs = epochs,
  verbose = 1,
  validation_data = list(
    data_val_x, data_val_y
  )
)

# history plot
plot(history)



data_test_pred <- model %>%
  predict_classes(data_test_x) %>%
  as.vector()



#  confusion matrix after model tuning

confmat_loan_tune <- confusionMatrix(factor(data_test_pred, labels = c("negative", "positive")), 
                                     factor(data_test$label, labels = c("negative", "positive")),
                                     mode = "prec_recall",
                                     positive = "negative")



eval_rf <- tidy(confmat_loan_tune) %>% 
  mutate(model = "Model1") %>% 
  select(model, term, estimate) %>% 
  filter(term %in% c("accuracy", "precision", "recall", "specificity"))

eval_rf %>% 
  write_csv(path = "data/conf_mat.csv")
