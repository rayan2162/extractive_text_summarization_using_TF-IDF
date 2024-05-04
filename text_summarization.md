# Text Summarization

## 0. Courses/learning

<https://huggingface.co/learn/nlp-course/en/chapter7/5>

## 1. Top NLP text summarization APIs and AI models

<https://medium.com/@pranathisoftwareservices/nlp-text-summarization-for-5-best-apis-ai-models-5b61534483a2>

### 1.1 Hugging Face's Transformers:

**BART (Bidirectional and Auto-Regressive Transformers):**

  - Hugging Face: 
    <https://huggingface.co/facebook/bart-large-cnn>
    
  - Text Summarization with BART Model:
    <https://medium.com/@sandyeep70/demystifying-text-summarization-with-deep-learning-ce08d99eda97>

**T5**

- T5-base fine-tuned fo News Summarization:
    <https://huggingface.co/mrm8488/t5-base-finetuned-summarize-news>

- T5 for text summarization in 7 lines of code:
    <https://medium.com/artificialis/t5-for-text-summarization-in-7-lines-of-code-b665c9e40771>

### 1.2 Sumy - Python library:

- A flexible python package called sumy allows for the developement of several text summarization methods, such as LEXRANK and LATENT SEMANTIC ANALYSIS(LSA)
- Extractive Text Summarization Techniques With sumy:
  <https://medium.com/@ondenyi.eric/extractive-text-summarization-techniques-with-sumy-3d3b127a0a32>

### 1.3 BRET summarizer:

- Well known for its semantic accuracy and bidirectional context awareness
- Frequently selected for summarization due to its contextual correctness and ability to capture subtle(very small diffrecne) in the input material.
- Has fine tuning flexibility
- But sequence length and Bret's Resource intensiveness(during fine tuning) must be taken to account.
- Implementing Extractive Text Summarization with BERT- [HCI + Applied AI Newsletter] Issue #4:
      
  <https://medium.com/@victor.dibia/implementing-extractive-text-summarization-with-bert-issue-4-e4856acb94ab>

---

## 2. Text Summarization for NLP: 5 Best APIs, AI Models, and AI Summarizers in 2024

<https://www.assemblyai.com/blog/text-summarization-nlp-5-best-apis/>

There are two methods in text summarization method
      
      1. Extractive
      2. Abstractive

### 2.1 Extractive Method

- This metod seek to extract the most pertinent information from a text.This is more traditional brcause of its simplicity.

- frequency based method will tend to rank the sentences in a text in order of importance by how frequently diffrent words are used. For each sentence there exists a weighting term for each word in the vocabulary,( where the weight is usually a function of the importacne of the word itself and the frequency with which the word appears throughout the document as a whole. )

- Graph based methods cast text document into the language of mathematical graph. Here every word is a node. Where nodes are connected if the sentences are deemed to be similar. Similar is constituded by diffrent algorithm and approach.For example one implimentation might use a threshhold on the **cosine similarity between TF-IDF vectors**. These **most similar** are considerd to have the most summarizing information. Notable graph based method is **TextRank** (version of googles page rank algorithm)

### 2.2 Abstractive Method 

- This method seek to generate a novel body of text that accurately summarizes the original text.
  
  There is a significant degree of freedom in not being limited to simply returning a subset of the original text.Despite their relative complexity abstract methods produces much more flexible and arguably faithful summaries.

  Deep learning **sec2sec** has proven extremely powerful.

---

## 3. Automatic Text Summarization with Machine Learning — An overview

<https://medium.com/luisfredgs/automatic-text-summarization-with-machine-learning-an-overview-68ded5717a25>

It is very challenging because when human summarizes a piece of text, we usually read it entirely to develope our understanding and then we write summary highliting its main points. Since computer lack human knowledge and language capability, it makes automatic text summarization a very difficult and non trivial task.

Most of approach model this problem as a classification problem which outputs weather to include a sentence in the summary or not. Other approaches have used **topic information**, **Latent Semantic Analysis (LSA)**, **Sequence to Sequence (s2s)**, **Reinforcement learning** and **Adversarial Process**.

### 3.1 The Extractive Approach

In this approach sentences directly form the document based on a scoring function.Here important section of text is identified and cropped out and stich together to produce condense version.

It is easier, yields naturally grammatical summarie that requires relatively little linguistic analysis.

 The 3 key steps of doing this:

    1. Construct an intermidieate representation. It works by computing TF metrics for each sentence in the given matrix.
   
    2. Scores the sentences based on the representation, assigining a vlaue to each sentences denoting the probability with which it will get picked up in the summary.
   
    3. Produces a summary based on the top K most important sentences. Some studies have used Latent Semantic Analysis(LSA) to identify semantically important sentences.

#### This paper proposed an extractive text summarization approach for fractual reports using deep learning model, exploring various features to improve the set of sequence selected for the summary

<https://arxiv.org/pdf/1708.04439>

#### A summarization framework based on convolutional neural network to learn sentences features and perform sentence ranking jointly using a CNN model for sentence ranking

<https://ieeexplore.ieee.org/abstract/document/7793761>

### 3.2 Abstractive Summarization

Abstractive summarization methods aim at producing summary by interpreting the text using advanced natural language techniques in order to generate a new shorter text — parts of which may not appear as part of the original document, that conveys the most critical information from the original text, requiring rephrasing sentences and incorporating information from full text to generate summaries such as a human-written abstract usually does.

#### A Neural Attention Model for Abstractive Sentence Summarization

<https://arxiv.org/pdf/1509.00685>

## 4. Text summarization using NLP

<https://medium.com/analytics-vidhya/text-summarization-using-nlp-3e85ad0c6349>

How to do text summarization

    1. Text cleaning
    2. Sentence tokenization
    3. Word tokenization
    4. Word-frequency table
    5. Summarization