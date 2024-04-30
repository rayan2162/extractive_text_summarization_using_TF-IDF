# Top NLP text summarization APIs and AI models

https://medium.com/@pranathisoftwareservices/nlp-text-summarization-for-5-best-apis-ai-models-5b61534483a2

**Hugging Face's Transformers:**
- Modern summarization models like BART and T5 are formidable pre-trained models 

**Sumy - Python library:**
- A flexible python package called sumy allows for the developement of several text summarization methods, such as LEXRANK and LATENT SEMANTIC ANALYSIS(LSA)

**BRET summarizer:**
- Well known for its semantic accuracy and bidirectional context awareness
- Frequently selected for summarization due to its contextual correctness and ability to capture subtle(very small diffrecne) in the input material.
- Has fine tuning flexibility
- But sequence length and Bret's Resource intensiveness(during fine tuning) must be taken to account.

---
# Text Summarization for NLP: 5 Best APIs, AI Models, and AI Summarizers in 2024

- There are two methods in text summarization method
  1. Extractive
  2. Abstractive

### Extractive Method
- This metod seel to extract the most pertinent information from a text.This is more traditional brcause of its simplicity.

This method work by identifying and extracting the salient information in a text.
For example 
- frequency based method will tend to rank the sentences in a text in order of importance by how frequently diffrent words are used. For each sentence there exists a weighting term for each word in the vocabulary,
    
     where the weight is usually a function of the importacne of the word itself and the frequency with which the word appears throughout the document as a whole. 

- Graph based methods cast text document into the language of mathematical graph. Here every word is a node. Where nodes are connected if the sentences are deemed to be similar. Similar is constituded by diffrent algorithm and approach.For example one implimentation might use a threshhold on the "cosine similarity between TF-IDF vectors". These "most similar" are considerd to have the most summarizing information. Notable graph based method is **TextRank** (version of googles page rank algorithm) 

### Abstractive Method
        
  - This method seek to generate a novel body of text that accurately summarizes the original text. 
  
    There is a significant degree of freedom in not being limited to simply returning a subset of the original text. The dificulty comes with an upside. Despite their relative complexity abstract methods produces much more flexible and arguably faithful summaries.
  
Deep learning sec2sec has proven extremely powerful. 
(RLHF - Reinforcement Learning form Human Feedback, here human feedback is used to train a reward model, which is then used to update an RL policy via PPO)