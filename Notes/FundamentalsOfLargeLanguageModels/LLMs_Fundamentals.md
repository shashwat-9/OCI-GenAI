# Large Language Models Fundamentals

 - Basics of LLMs - What LLMs are? What they do? How they work?
 - Prompting Techniques - Techniques to generate text with specific characteristics
 - Training and Decoding - Technical terms for generating texts with LLMs
 - Dangers of LLMs based technology deployment - Includes things like prompt injection and hallucination
 - Upcoming cutting Edge topics discussion

## Introduction
 - A language model(LM) is a probabilistic model of text. It computes a distribution over a vocabulary.
 - The LM gives a probability to every word in its vocabulary of appearing in the blank.

```
    I wrote to a zoo to send me a pet. They sent me a ________
    
    word        lion elephant dog cat panther alligator
    Probabilty  0.1     0.1   0.3 0.2 0.05      0.02
    
    - What a language model will compute for us here is a distribution over a vocabulary.
    - The language model knows about a set of words called vocabulary and the LM will assign a probabilty to each of
    those words in its vocabulary for each of the word appearing in the blank.
    -  When we run a sequnece of words through a LM, we get a probability for each of the words in its vocabulary
```

 - Large in "Large Language model"(LLM) refers to the # of parameters; no agreed upon threshold.
 - LLMs are no different to simple LMs.

## This module
 - LLM architecture
 - Prompt and Training - How do we affect the distribution over the vocabulary?
 - Decoding - How do LLMs generate text using these distributions?


## LLM Architectures
Two major architectures for LMs :
    1. Encoders
    2. Decoders
 - Multiple architectures focused on encoding and decoding, i.e. embedding and text generation
 - All Models built on the Transformer Architecture
 - Each type of model has different Capabilities(embedding/generation)
 - Embedding text generally means converting a sequence of words into a single vector or sequence of vector.
 - In other word, embedding is a numeric representation of text that tries to capture the meaning of the text.
 - Decoders models are designed to decode or generate text
 - Models of each type come in a variety of sizes(# of parameters). Sizes in context of Models is number of trainable 
parameters.

![Model Ontology](./ModelOntology.png)
 - There is no reason why we couldn't build a large encoder, it's just that we don't need to.
 - When models are too small, they tend to be poor text generators. However, with advanced techniques, it may be possible
to make better text generators with smaller models.

## Encoders
 - Models that convert a sequence of words to an embedding(vector representation).
![Embedding](./EmbeddingByEncoders.png)
 - Example includes MiniLM, Embed-light, BERT, RoBERTA, DistillBERT, SBERT

## Decoders
 - models take a sequence of words and output next word
 - Example includes GPT-4, Llama, BLOOM, Falcon, ...
 - Decoders generate the next token in the input sequence based on the vocabulary which they compute.
 - Decoder only produce a single token at a time. So, if we want some text generated, we append the output token in the
sequence provided, and again invoke the decoder. Though it is computationally expensive.
 - Decoders shouldn't be used for embedding.
![Decoders](./GenerationsByDecoders.png)

## Encoders-Decoders
 - Encoders-decoders - encodes a sequence of words and use the encoding + to output a next word.
 - Examples includes - T5, UL2, BART, ...
 - The below image is for a translator.
![Encoders&Decoders](./Encoders&Decoders.png)
 - Not all models are suitable for all tasks. The below image gives an overview on this :
![ArchitectureAtGlance](./ArchitectureAtGlance.png)


# Prompting and Prompt Engineering
 - Here onwards, only decoder will be discussed. When we talk about LLMs, we generally refer to Decoder models.
```
    I wrote to the zoo to send me a pet. They sent me a ______
    
    word        lion    elephant    dog     cat     panther     alligator
    Probability  0.1    0.1         0.3     0.2     0.05        0.02
```
 - A LLM computes the distribution over the next word in the sequence
 - To exert some control over the LLM, we can affect the probability over vocabulary in 2 ways.
 1. Prompting
 2. Training

 - The simplest way to affect the distribution over the vocabulary is to change the prompt.
 - __Prompt__ -> the text provided to an LLM as input, sometimes containing instructions and/or examples
 - The distribution changes when some other word is added to the same sequence :
 - Very Large decoder only model is trained in a procedure called pre-training. During this training, the model is fed
  tremendous amount of text, that is typically quite varied.

```
    - Just adding the word 'Little' at the end of the prompt, changes the distribution over vocabulary
    I wrote to the zoo to send me a pet. They sent me a ______
    
    word        lion    elephant    dog     cat     panther     alligator
    Probability  0.03    0.02       0.45    0.4     0.05        0.01

     - During pretraining, the model knows which animal is bigger and which one is smaller. As it would have encountered
 ```

## Prompt Engineering
 - __Prompt Engineering__ -> the process of iteratively refining a prompt for the purpose of eliciting a particular 
style of response.
 - In layman's term, By changing the input we get different results.
 - Prompt engineering is challenging, often unintuitive, and not guaranteed to work. Even a space can alter the 
probability distribution.
 - At the same time, it can be effective; multiple tested prompt-design strategies exist.

### In-Context Learning and Few-shot Prompting
 - In-context learning - conditioning(prompting) an LLM with instructions and or demonstrations of the task it is meant
to complete. Here, none of the parameter is changing, instead it refers to constructing a prompt that demonstrate the 
tasks that the model should complete.

 - k-shot prompting - explicitly providing k examples of the intended task in the prompt
![3shotPrompting.png](./3shotPrompting.png)
 - This example is from GPT-3 model. This is a 3-shot prompting, as there are 3 examples.
 - _prompt_ word is generally overloaded. Here in the above paper, the last line is the prompt.
 - __Few-shot prompting is widely believed to improve results over 0-shot prompting__
![ExampleOfPrompts](./ExampleOfDIfferentPrompts.png)

 - If we give a big problem at once, and ask the answer, it might fail, as the decoder is generating one word at a time.
 - Rather, we should break it into smaller problems while prompting, so that the next word generated and thus the answer
is correct. The one example is chain of thought and LeastToMost prompting given below.
![ChainOfThought](./ChainOfThought.png)
 - The Least To Most prompting works better than teh chain of thought. This prompting, increasingly solves the problem.
![LeastToMost](./LeastToMost.png)
 - Step-Back - prompt the LLM to identify high-level concepts pertinent to a specific task. It is observed that if
asking a question of physics and chemistry, if the related equations and equations are emitted first, the results are
accurate.
![Step-Back](./Step-Back.png)

## Issues with Prompting
Here we will discuss, how prompting can be used to elicit unintended or even harmful behaviour from a model.
 - Prompt Injection(jailBreaking) - to deliberately provide an LLM with input that attempts to cause it to ignore 
instructions, cause harm, or behave contrary to deployment expectations.
![PromptInjection](./PromptInjection.png)
 - Due to these and many such attacks, the LLMs must be safeguarded, so that unnecessary details doesn't gets leaked out.

## Training
 - Most significant way to alter the distribution over a vocabulary is training.
 - Prompting alone may be inappropriate when: training data exits, or domain adaption is required. With prompting, we're
limited with the extent we can change the distribution over vocabulary.
 - Domain-adaption -> adapting a model(typically via training) to enhance its performance outside the domain/subject-area
it was trained on.
 - There are various ways to train to a LLM :
    1. Fine-tuning(FT) -> In this training, we modify all the parameters, using a labelled data set for a specific task,
which alters all the parameters of the LLM.
    2. Param. Efficient ET -> Here we take a small subset of the parameters to train or add a set of new parameters to 
train the LLM using Labelled, task-specific data set. It is cost-efficient, w.r.t current really large models, as less 
training is required.
    3. Soft prompting -> It is another cheap prompting option. Add parameters to the prompt, some specialized word that 
can be used for training.
    4. Continual pre-training -> It is same as fine-tuning but the data fed to the model is random and un-labelled, and
the model is supposed to predict the next word continually.
![TrainingTechniques](./TrainingTechniques.png)

### Cost of Training
![CostOfTraining](./CostOfTraining.png)


## Decoding
 - Decoding is the process of generating text with LLM.
```
    I wrote to the zoo to send me a pet. They sent me a ______
    word        lion    elephant    dog     cat     panther     alligator
    probaility  0.03    0.02        0.45    0.4     0.05        0.01
```
 - Decoding happens iteratively, 1 word at a time.
 - At each step of decoding, we use the distribution over vocabulary and select 1 word to emit.
 - The word is appended to the input, and we feed the revised prompt to the LLM, and then the decoding process continues
 - There are various ways to select a word out of the vocabulary distribution.
    1. Greedy Decoding 
    2. Nucleus sampling -> Defines what portion of the distribution over words you're allowed to sample from
    3. Beam search -> Where we simultaneously generate multiple similar sequences and continually prune the sequences with low probability.
### Greedy Decoding
 - Picks the highest probability word at each step
 - Naive yet effective method
![GreedyDecoding](./GreedyDecoding.png)

### Non-Deterministic Decoding
 - Picks randomly among a high probability candidate at each step.
![NonDeterministicDecoding](./NonDeterministicDecoding.png)

#### Temperature
 - When decoding temperature is a (hyper) parameter that modulates the distribution over vocabulary.
 - When temperature is increased, the distribution is flattened, i.e. probability of every word comes closer to every other.
 - When temperature is decreased, the distribution is more peaked around the most likely word. The most likely word gets
the highest probability.
 - The temperature changes doesn't the order of distribution in the vocabulary, i.e. the lowest or the highest probability
words will stay the lowest and highest respectively. __The relative ordering of words is unaffected by temperature__
 - With sampling on, increasing temperature makes the model deviate more from greedy decoding

## Hallucination
 - Hallucination -> generated text that is non-factual and/or ungrounded.
 - There are some methods that are claimed to reduce hallucinations(e.g. retrieval-augmentation)
 - There is no known methodology to reliably keep LLMs from hallucinating.

### Groundedness and Attributability
 - Grounded -> generated text is grounded in a document if the document supports the text
 - The method works by comparing the text generated by LLMs with a candidate document
 - The research community has embraced attribution/grounding
 - Attributed QA, system must output a document that grounds its answer
 - The TRUE model: for measuring groundedness via NLI
 - Train an LLM to output sentences with citations

## LLM Applications

### Retrieval Augmented Generation (RAG)
 - Primarily used in Q&A, where the model has access to (retrieved) support documents for query.
1 . Input -> 2. Corpus -> 3. LLM
 - The flow is, the user enters an input, as a question, that maybe used to search for related documents called
corpus, which is returned to the LLM, that will be fed to the LLM along with the question to generate sensible answer.
 - Claimed to reduce hallucination
 - Multi-document Q&A via fancy decoding, e.g. RAG-tok
 - Idea has gotten a lot of traction
    1. Used in dialogue, Q&A, fact-checking, slot filling, entity-linking
    2. Non-parametric; in theory, the same model can answer questions about any corpus. For answering question with a 
        new subject, we can just provide the relevant document as the corpus that can be searched for answering the 
        question.
    3. Can be trained end-to-end.

### Code Models
 - Instead of training on written language, they're trained on code, documentation and comments
 - Co-pilot, Codex, Code Llama
 - Complete partly written functions, synthesize programs from docstring, debugging
 - Largely successful > 85% of people using co-pilot feel more productive
 - Great fit between training data(code + comments) and test-time tasks(write code + comments). 
Also, code is structured -> easier to learn.
 - This is unlike LLMs, which can be trained on a wide variety of internet text and used for many purposes(other than
generating internet text); code models have (arguably) narrower scope.

### Multi-Modal
 - These are models trained on multiple modalities, e.g. language and images
 - Models can be autoregressive, e.g., DALL-E or diffusion-based e.g., stable Diffusion
 - Diffusion-models can produce a complex output simultaneously, rather than token-by-token
    1. Difficult to apply to text because text is categorical.
    2. Some attempts have been made; still not very popular.
 - These models can perform either image-to-text, text-to-image tasks(or both), video generations, audio generation
 - Recent retrieval-augmentation extensions

### Language Agents
 - Language agents are models that are intended for sequential decision-making scenarios. For e.g., playing chess,
operating software system autonomously or searching a web page for question.
 - A budding area of research where LLM-based agents
    1. create plans and "reason"
    2. Take actions in response to plans and the environment
    3. Are capable of using tools
 - Some notable work in this space: 
    1. ReAct - Iterative framework where LLM emits thoughts, then acts, and observe result.
    2. Toolformer - Pre-training technique where strings are replaced with calls to tools that yield result.
    3. Bootstrapped reasoning - Prompt the LLM to emit rationalization of intermediate steps; use as fine-tuning data.
