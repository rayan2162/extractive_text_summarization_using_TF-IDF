# Sentence Autocomplete

## Building an Autocomplete System

<https://thegrigorian.medium.com/building-an-autocomplete-system-bb3392284a5e>

1. Data Collection and Preprocessing
   - We start with data collection and preprocessing steps, such as cleaning and tokenization.
2. Building Language Models
    - Common language models include **n-gram models**, **Markov models**, and more advanced methods like **recurrent neural networks (RNNs)** or **transformers**.
3. Generating Suggestions
    - the N-gram language model predicts the next word based on the N-1 preceding words. For instance, if the user types “The cat is,” the model predicts the most likely next word, such as “sleeping,” “playing,” or “running,” based on the probabilities it has learned from the training data.
4. Ranking and Filtering
    - This could involve techniques such as sorting by probability, applying heuristics, or employing more advanced methods like beam search
5. Evaluation and Iteration
    - Building a functional autocomplete system is an iterative process. By evaluating the performance of the system you can suggest ways to fine-tune and improve it based on user feedback and real-world usage.

---

## Sentence Autocomplete Using Pytorch

<https://www.geeksforgeeks.org/sentence-autocomplete-using-pytorch/>

    1. Cleaning the text data for training the NLP model
    2. Loading the dataset using PyTorch
    3. Creating the LSTM model
    4. Training an NLP model
    5. Making inferences from the trained model

---

## Sentence Autocomplete Using TensorFlow from Scratch

<https://www.geeksforgeeks.org/sentence-autocomplete-using-tensorflow-from-scratch/>

1. Creating the model for Sentence Autocompletion
        Step 1: Importing necessary libraries
        Step 2: Loading the dataset
        Step 3: Extracting text from the dataset
        Step 4: Cleaning Text
        Step 5: Text vectorization and One hot encoding
        Step 6: Building the model
        Step 7: Compiling and Training the Model
        Step 8: Sentence Autocomplete
2. Creating the Flask website
3. Step by Step Process for Deploying the project using Docker

---

## N-gram

<https://deepai.org/machine-learning-glossary-and-terms/n-gram>

N-grams are contiguous sequences of n items from a given sample of text or speech. The items can be phonemes, syllables, letters, words, or base pairs according to the application.

The concept of an n-gram is straightforward: it is a sequence of 'n' consecutive items. For instance, in the domain of text analysis, if 'n' is 1, we call it a unigram; if 'n' is 2, it is a bigram; if 'n' is 3, it is a trigram, and so on. The larger the value of 'n', the more context you have, but with diminishing returns on information gained versus computational expense and data sparsity.

Consider the sentence "The quick brown fox jumps over the lazy dog." Here are some examples of n-grams derived from this sentence:

    Unigrams: "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"
    Bigrams: "The quick", "quick brown", "brown fox", "fox jumps", "jumps over", "over the", "the lazy", "lazy dog"
    Trigrams: "The quick brown", "quick brown fox", "brown fox jumps", "fox jumps over", "jumps over the", "over the lazy", "the lazy dog"

As you can see, unigrams do not contain any context, bigrams contain a minimal context, and trigrams start to form more coherent and contextually relevant phrases.

- N-gram language models have limitations, particularly when dealing with longer contexts. They may struggle to capture complex language patterns and relationships that span beyond a few words. Additionally, they might face challenges when handling unseen or rare N-grams, leading to inaccurate predictions.

### Challenges with N-Grams

While n-grams are a powerful tool, they come with their own set of challenges:

    Data Sparsity: As 'n' increases, the frequency of specific n-gram sequences decreases, leading to data sparsity issues. This can make it difficult for models to learn effectively from such sparse data.

    Computational Complexity: The number of possible n-grams increases exponentially with 'n', which can lead to increased computational costs and memory requirements.

    Context Limitation: N-grams have a fixed context window of 'n' items, which may not be sufficient to capture longer dependencies in the text.

### Smoothing Techniques

To address data sparsity and improve the performance of n-gram models, various smoothing techniques are employed. These techniques adjust the probability distribution of n-grams to account for unseen or rare n-gram sequences. Some common smoothing techniques include: **Additive or Laplace smoothing**, **Good-Turing discounting**,**Backoff and interpolation methods**.

---

## Introduction to N-gram

<https://youtu.be/hM49MPmakNI>

- Goal: Compute the probability of a sentence or sequence of word
p(w)=p(w1, w2, w3, ...wn)
- Related task: Probability if an upcoming word.
p(w5 | w1, w2, w3, w4)
- A model that computes either of these is called a language model(is acuallly grammer because like grammer language model also determines which comes after which)

For example:
P( its, water, is, so, transparent, that )

intuition: Chain rule of probability.

**Chain rule: P(a, b, c, d) = P(a) P(b|a) P(c|a,b) P(d|a,b,c)**

P(its) x P(water|its) x P(is|its water) x P(so|its water is) x P(transparent|its water is so)

P(the| its water is so transparent that) = COUNT(its water is so transparent that the) / COUNT(its water is so transparent that)

This isnt possible due to too many possible sentences.

So, we use **Merkov Assumption**

p(the| its water is so transparent that) instead of this we use P(the|that).
or
P(the| its water is so transparent that) instead of this we use P(the|transparent that)
