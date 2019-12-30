 # Documentation of RASA Framework


## Steps in creating a RASA Project

1. Create a New Project
2. View Your NLU Training Data
3. Define Your Model Configuration
4. Write Your First Stories
5. Define a Domain
6. Train a Model
7. Talk to Your Assistant


## 1. Create a New Project

The first step is to create a new Rasa project. To do this, run:

``` rasa init --no-prompt ```


This creates the following files:

### Files present in RASA 

| __init__.py               | an empty file that helps python find your actions    |
|---------------------------|------------------------------------------------------|
| actions.py                | code for your custom actions                         |
| config.yml            | configuration of your NLU and Core models            |
| credentials.yml           | details for connecting to other services             |
| data/nlu.md            | your NLU training data                               |
| data/stories.md        | your stories                                         |
| domain.yml             | your assistant’s domain                              |
| endpoints.yml             | details for connecting to channels like fb messenger |
| models/<timestamp>.tar.gz | your initial model                                   |



## 2. View Your NLU Training Data

The first piece of a Rasa assistant is an NLU model. NLU stands for Natural Language Understanding, which means turning user messages into structured data. To do this with Rasa, you provide training examples that show how Rasa should understand user messages, and then train a model by showing it those examples.

Run the code cell below to see the NLU training data:


``` cat data/nlu.md ```


## 3. Define Your Model Configuration

The configuration file defines the NLU and Core components that your model will use. 

``` cat config.yml ```


## 4. Write Your First Stories

At this stage, you will teach your assistant how to respond to your messages. This is called dialogue management, and is handled by your Core model.

Core models learn from real conversational data in the form of training “stories”. A story is a real conversation between a user and an assistant. Lines with intents and entities reflect the user’s input and action names show what the assistant should do in response.

Run the command below to view the example stories inside the file data/stories.md:

``` cat data/stories.md ```


## 5. Define a Domain

The next thing we need to do is define a Domain. The domain defines the universe your assistant lives in: what user inputs it should expect to get, what actions it should be able to predict, how to respond, and what information to store. The domain for our assistant is saved in a file called domain.yml:

``` cat domain.yml ```


## 6. Train a Model

Anytime we add new NLU or Core data, or update the domain or configuration, we need to re-train a neural network on our example stories and NLU data. To do this, run the command below. This command will call the Rasa Core and NLU train functions and store the trained model into the models/ directory. The command will automatically only retrain the different model parts if something has changed in their data or configuration.

``` rasa train ```
 
 
## 7. Talk to Your Assistant

``` rasa shell   ```
