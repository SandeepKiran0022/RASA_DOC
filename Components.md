
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
	    
