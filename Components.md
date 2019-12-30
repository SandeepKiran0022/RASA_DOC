
## Pipeline in RASA Framework

- name: "WhitespaceTokenizer"
- name: "CRFEntityExtractor"
- name: "EntitySynonymMapper"
- name: "CountVectorsFeaturizer"
- name: "EmbeddingIntentClassifier"


## State of the ART Models

There are currently many state of the art models for NLP, but these have proven to be efficient.

- BERT
- RoBERTa
- DistilBERT
- XLNET


![alt text](https://github.com/SandeepKiran0022/RASA_DOC/blob/master/performance.png)

![alt_test](https://github.com/SandeepKiran0022/RASA_DOC/blob/master/perf2.png)


## So which one to use?
 
If you really need a faster inference speed but can compromise few-% on prediction metrics, DistilBERT is a starting reasonable choice, however, if you are looking for the best prediction metrics, you’ll be better off with Facebook’s RoBERTa.

Theoratically, XLNet’s permutation based training should handle dependencies well, and might work better in longer-run.

However, Google’s BERT does serve a good baseline to work with and if you don't have any of the above critical needs, you can keep your systems running with BERT.

 
## Thus, for most of the components BERT will be used as a State of the Art Model.


## BERT

### How BERT works?

- BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, Transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since BERT’s goal is to generate a language model, only the encoder mechanism is necessary. The detailed workings of Transformer are described in a paper by Google.

- As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

- The chart below is a high-level description of the Transformer encoder. The input is a sequence of tokens, which are first embedded into vectors and then processed in the neural network. The output is a sequence of vectors of size H, in which each vector corresponds to an input token with the same index.

- When training language models, there is a challenge of defining a prediction goal. Many models predict the next word in a sequence (e.g. “The child came home from ___”), a directional approach which inherently limits context learning. To overcome this challenge, BERT uses two training strategies:

- Masked LM (MLM)

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. In technical terms, the prediction of the output words requires:
Adding a classification layer on top of the encoder output.


Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
Calculating the probability of each word in the vocabulary with softmax.


![alt_text](https://github.com/SandeepKiran0022/RASA_DOC/blob/master/pic3.png)

The BERT loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words. As a consequence, the model converges slower than directional models, a characteristic which is offset by its increased context awareness.

- Next Sentence Prediction (NSP)


In the BERT training process, the model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document. During training, 50% of the inputs are a pair in which the second sentence is the subsequent sentence in the original document, while in the other 50% a random sentence from the corpus is chosen as the second sentence. The assumption is that the random sentence will be disconnected from the first sentence.

To help the model distinguish between the two sentences in training, the input is processed in the following way before entering the model:

- A [CLS] token is inserted at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.

- A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar in concept to token embeddings with a vocabulary of 2.

- A positional embedding is added to each token to indicate its position in the sequence. The concept and implementation of positional embedding are presented in the Transformer paper.


![alt_text](https://github.com/SandeepKiran0022/RASA_DOC/blob/master/pic4.png)


To predict if the second sentence is indeed connected to the first, the following steps are performed:

- The entire input sequence goes through the Transformer model.

- The output of the [CLS] token is transformed into a 2×1 shaped vector, using a simple classification layer (learned matrices of weights and biases).

- Calculating the probability of IsNextSequence with softmax.
When training the BERT model, Masked LM and Next Sentence Prediction are trained together, with the goal of minimizing the combined loss function of the two strategies.

### Different Tokenization components provided by RASA Framework

- Jieba Tokenizer
- Mitie Tokenizer
- Spacy Tokenizer
- Whitespace Tokenizer


### - Whitespace Tokenizer
	
Tokenizer using whitespaces as a separator

Description:	
Creates a token for every whitespace separated character sequence. Can be used to define tokens for the MITIE entity extractor.

Configuration:	
If you want to split intents into multiple labels, e.g. for predicting multiple intents or for modeling hierarchical intent structure, use these flags:

- tokenization of intent and response labels:
intent_split_symbol sets the delimiter string to split the intent and response labels, default is whitespace.
Make the tokenizer not case sensitive by adding the case_sensitive: false option. Default being case_sensitive: true.

``` 
pipeline:
- name: "WhitespaceTokenizer"
  case_sensitive: false
  
  ```
### - Jieba Tokenizer

	
Tokenizer using Jieba for Chinese language

Description:	
Creates tokens using the Jieba tokenizer specifically for Chinese language. For language other than Chinese, Jieba will work as WhitespaceTokenizer. Can be used to define tokens for the MITIE entity extractor. Make sure to install Jieba, pip install jieba.

Configuration:	
User’s custom dictionary files can be auto loaded by specific the files’ directory path via dictionary_path

```
pipeline:
- name: "JiebaTokenizer"
  dictionary_path: "path/to/custom/dictionary/dir"
```

If the dictionary_path is None (the default), then no custom dictionary will be used.


### - MitieTokenizer
	
MitieNLP

Description:	
Creates tokens using the MITIE tokenizer. Can be used to define tokens for the MITIE entity extractor.

Configuration:

```
pipeline:
- name: "MitieTokenizer"
```
### - Spacy Tokenizer

Tokenizer using spacy

Description:	Creates tokens using the spacy tokenizer. Can be used to define tokens for the MITIE entity extractor.


## - Evaluvation of Best Tokenizer available in RASA

- We can see that Whitespace tokenizer is the best and efficient option for tokenization in RASA ,if the language is English or any other language which uses space as a separator between words.

- If the language is Chinese , then using Jieba Tokenizer is the best option

### - Whitespace tokenizer ( tokenization function)

``` 
 words = re.sub(
                # there is a space or an end of a string after it
                r"[^\w#@&]+(?=\s|$)|"
                # there is a space or beginning of a string before it
                # not followed by a number
                r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
                # not in between numbers and not . or @ or & or - or #
                # e.g. 10'000.00 or blabla@gmail.com
                # and not url characters
                r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
                " ",
                text,
            ).split()

```

### - State of the ART Model for tokenization





