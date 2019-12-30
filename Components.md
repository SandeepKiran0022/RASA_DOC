
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

## - Masked LM (MLM)

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. In technical terms, the prediction of the output words requires:
Adding a classification layer on top of the encoder output.


Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
Calculating the probability of each word in the vocabulary with softmax.


![alt_text](https://github.com/SandeepKiran0022/RASA_DOC/blob/master/pic3.png)

The BERT loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words. As a consequence, the model converges slower than directional models, a characteristic which is offset by its increased context awareness.

## - Next Sentence Prediction (NSP)


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


## Takeaways

- Model size matters, even at huge scale. BERT_large, with 345 million parameters, is the largest model of its kind. It is demonstrably superior on small-scale tasks to BERT_base, which uses the same architecture with “only” 110 million parameters.

- With enough training data, more training steps == higher accuracy. For instance, on the MNLI task, the BERT_base accuracy improves by 1.0% when trained on 1M steps (128,000 words batch size) compared to 500K steps with the same batch size.

- BERT’s bidirectional approach (MLM) converges slower than left-to-right approaches (because only 15% of words are predicted in each batch) but bidirectional training still outperforms left-to-right training after a small number of pre-training steps.


## Tokenization

- Tokenization is the process of tokenizing or splitting a string, text into a list of tokens. One can think of token as parts like a word is a token in a sentence, and a sentence is a token in a paragraph.

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

- We can see that ``` Whitespace tokenizer``` is the best and efficient option for tokenization in RASA ,if the language is English or any other language which uses space as a separator between words.

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

- We can see that BERT provides tokenization , but generally ``` White space tokenizer ``` is efficient than BERT's tokenization as a service.

- BERT Tokenization as a service ( sample code)

Often you want to use your own tokenizer to segment sentences instead of the default one from BERT. Simply call encode(is_tokenized=True) on the client slide as follows:

```
texts = ['hello world!', 'good day']

# a naive whitespace tokenizer
texts2 = [s.split() for s in texts]

vecs = bc.encode(texts2, is_tokenized=True)

``` 

This gives [2, 25, 768] tensor where the first [1, 25, 768] corresponds to the token-level encoding of “hello world!”. If you look into its values, you will find that only the first four elements, i.e. [1, 0:3, 768] have values, all the others are zeros. 

This is due to the fact that BERT considers “hello world!” as four tokens: [CLS] hello world! [SEP], the rest are padding symbols and are masked out before output.


## Entity Extraction

- Entity extraction, also known as entity name extraction or named entity recognition, is an information extraction technique that refers to the process of identifying and classifying key elements from text into pre-defined categories. In this way, it helps transform unstructured data to data that is structured, and therefore machine readable and available for standard processing that can be applied for retrieving information, extracting facts and question answering. 


## Different Entity extraction component provided by RASA 

- MitieEntityExtractor
- SpacyEntityExtractor
- EntitySynonymMapper
- CRFEntityExtractor
- DucklingHTTPExtractor


### MitieEntityExtractor

- MITIE entity extraction (using a MITIE NER trainer)

- appends entities

- Requires:MitieNLP

- Output_Example:
```
{
    "entities": [{"value": "New York City",
                  "start": 20,
                  "end": 33,
                  "confidence": null,
                  "entity": "city",
                  "extractor": "MitieEntityExtractor"}]
}
```
- This uses the MITIE entity extraction to find entities in a message. The underlying classifier is using a multi class linear SVM with a sparse linear kernel and custom features. The MITIE component does not provide entity confidence values.

- Configuration:	
```
pipeline:
- name: "MitieEntityExtractor"
```

### SpacyEntityExtractor

- appends entities

- Requires: SpacyNLP

- Output_Example:	
```
{
    "entities": [{"value": "New York City",
                  "start": 20,
                  "end": 33,
                  "entity": "city",
                  "confidence": null,
                  "extractor": "SpacyEntityExtractor"}]
}
```

-Using spaCy this component predicts the entities of a message. spacy uses a statistical BILOU transition model. As of now, this component can only use the spacy builtin entity extraction models and can not be retrained. This extractor does not provide any confidence scores.

-Configuration:	

Configure which dimensions, i.e. entity types, the spacy component should extract. A full list of available dimensions can be found in the spaCy documentation. Leaving the dimensions option unspecified will extract all available dimensions.

- pipeline:
```
- name: "SpacyEntityExtractor"
  # dimensions to extract
  dimensions: ["PERSON", "LOC", "ORG", "PRODUCT"]
  ```
  
### CRFEntityExtractor

-conditional random field entity extraction

-Requires A tokenizer

-Output-Example:	
```
{
    "entities": [{"value":"New York City",
                  "start": 20,
                  "end": 33,
                  "entity": "city",
                  "confidence": 0.874,
                  "extractor": "CRFEntityExtractor"}]
}
```

-Description:	

This component implements conditional random fields to do named entity recognition. CRFs can be thought of as an undirected Markov chain where the time steps are words and the states are entity classes. Features of the words (capitalisation, POS tagging, etc.) give probabilities to certain entity classes, as are transitions between neighbouring entity tags: the most likely set of tags is then calculated and returned. If POS features are used (pos or pos2), spaCy has to be installed. If you want to use additional features, such as pre-trained word embeddings, from any provided dense featurizer, use "text_dense_features". Make sure to set "return_sequence" to True in the corresponding featurizer.

Configuration:	

- pipeline:
```
- name: "CRFEntityExtractor"
  # The features are a ``[before, word, after]`` array with
  # before, word, after holding keys about which
  # features to use for each word, for example, ``"title"``
  # in array before will have the feature
  # "is the preceding word in title case?".
  # Available features are:
  # ``low``, ``title``, ``suffix5``, ``suffix3``, ``suffix2``,
  # ``suffix1``, ``pos``, ``pos2``, ``prefix5``, ``prefix2``,
  # ``bias``, ``upper``, ``digit``, ``pattern``, and ``text_dense_features``
  features: [["low", "title"], ["bias", "suffix3"], ["upper", "pos", "pos2"]]

  # The flag determines whether to use BILOU tagging or not. BILOU
  # tagging is more rigorous however
  # requires more examples per entity. Rule of thumb: use only
  # if more than 100 examples per entity.
  BILOU_flag: true

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  max_iterations: 50

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  # Specifies the L1 regularization coefficient.
  L1_c: 0.1

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  # Specifies the L2 regularization coefficient.
  L2_c: 0.1
  
```

### DucklingHTTPExtractor

-Duckling lets you extract common entities like dates, amounts of money, distances, and others in a number of languages.

-Output-Example:	

```
{
    "entities": [{"end": 53,
                  "entity": "time",
                  "start": 48,
                  "value": "2017-04-10T00:00:00.000+02:00",
                  "confidence": 1.0,
                  "extractor": "DucklingHTTPExtractor"}]
}
```

-To use this component you need to run a duckling server. The easiest option is to spin up a docker container using docker run -p 8000:8000 rasa/duckling.

- Alternatively, you can install duckling directly on your machine and start the server.

-Duckling allows to recognize dates, numbers, distances and other structured entities and normalizes them. Please be aware that duckling tries to extract as many entity types as possible without providing a ranking. For example, if you specify both number and time as dimensions for the duckling component, the component will extract two entities: 10 as a number and in 10 minutes as a time from the text I will be there in 10 minutes. In such a situation, your application would have to decide which entity type is be the correct one. The extractor will always return 1.0 as a confidence, as it is a rule based system.

-Configure which dimensions, i.e. entity types, the duckling component should extract. A full list of available dimensions can be found in the duckling documentation. Leaving the dimensions option unspecified will extract all available dimensions.

- pipeline:
```
- name: "DucklingHTTPExtractor"
  # url of the running duckling server
  url: "http://localhost:8000"
  # dimensions to extract
  dimensions: ["time", "number", "amount-of-money", "distance"]
  # allows you to configure the locale, by default the language is
  # used
  locale: "de_DE"
  # if not set the default timezone of Duckling is going to be used
  # needed to calculate dates from relative expressions like "tomorrow"
  timezone: "Europe/Berlin"
  # Timeout for receiving response from http url of the running duckling server
  # if not set the default timeout of duckling http url is set to 3 seconds.
  timeout : 3  
```

## Evaluvation of best component for Entity extraction in RASA 

- If we have custom trained entities , then Conditional Random Field ( CRF) Entity extraction is the best.
- If we dont have custom trained entities, then Duckling Http Extractr is the best.

- Thus, CRF Entity extraction is the best component for entity extraction in RASA as for banking chatbots we dont required pre trained entities.

![alt_text](https://github.com/SandeepKiran0022/RASA_DOC/blob/master/pic5.png)


## State of the ART Models for Entity Extraction
