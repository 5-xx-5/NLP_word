# Word Prediction using Convolutional Neural Networks—can you do better than iPhone™ Keyboard?

In this project, we examine how well neural networks can predict the current or next word. Language modeling is one of the most important nlp tasks, and you can easily find deep learning approaches to it. Our contribution is threefold. First, we want to make a model that simulates a mobile environment, rather than having general modeling purposes. Therefore, instead of assessing perplexity, we try to save the keystrokes that the user need to type. Next, we use CNNs instead of RNNs, which are more widely used in language modeling tasks. RNNs—even improved types such as LSTM or GRU—suffer from short term memory. Deep layers of CNNs are expected to overcome the limitation. Finally, we employ a character-to-word model here. Concretely, we predict the current or next word, seeing the preceding 50 characters. Because we need to make a prediction at every time step of typing, the word-to-word model dont't fit well. And the char-to-char model has limitations in that it depends on the autoregressive assumption. 

## Requirements
  * numpy >= 1.11.1
  * sugartensor >= 0.0.2.4
  * lxml >= 3.6.4.
  * nltk >= 3.2.1.
  * regex

## Background / Glossary / Metric

<img src="image/word_prediction.gif" width="200" align="right">


* Full Keystrokes (FK): the keystrokes when supposing that the user has deactivated the prediction option. In this exeriment, the number of FK is the same as the number of characters (including spaces).
* Responsive Keystroke (RK): the keystrokes when supposing that so the user always choose it if their intended word is suggested. Especially, we take only the top candidate into consideration here. 
* Keystroke Savings Rate (KSR): the rate of savings by a predictive engine. It is simply calculated as follows.
  * KSR = (FK - RK) / FK 


## Data
* For training and test, we build an English news corpus from wikinews dumps for the last 6 months.

## Model Architecture / Hyper-parameters

* 20 * conv layer with kernel size=5, dimensions=300
* residual connection

## Work Flow

* STEP 1. Download [English wikinews dumps](https://dumps.wikimedia.org/enwikinews/20170120/).
* STEP 2. Extract them and copy the xml files to `data/raw` folder.
* STEP 3. Run `build_corpus.py` to build an English news corpus.
* STEP 4. Run `prepro.py` to make vocabulary and training/test data.
* STEP 5. Run `train.py`.
* STEP 6. Run `eval.py` to get the results for the test sentences.
* STEP 7. We manually tested for the same test sentences with iPhone 7 keyboard.

