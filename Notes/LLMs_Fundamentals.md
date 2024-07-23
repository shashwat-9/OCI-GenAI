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

![Model Ontology](/ModelOntology.png)
 - There is no reason why we couldn't build a large encoder, it's just that we don't need to.
 - When models are too small, they tend to be poor text generators. However, with advanced techniques, it may be possible
to make better text generators with smaller models.

## Encoders
 - Models that convert a sequence of words to an embedding(vector representation).
![Embedding](/EmbeddingByEncoders.png)
 - Example includes MiniLM, Embed-light, BERT, RoBERTA, DistillBERT, SBERT

## Decoders
 - models take a sequence of words and output next word
 - Example includes GPT-4, Llama, BLOOM, Falcon, ...
 - Decoders generate the next token in the input sequence based on the vocabulary which they compute.
 - Decoder only produce a single token at a time. So, if we want some text generated, we append the output token in the
sequence provided, and again invoke the decoder. Though it is computationally expensive.
 - Examples include GPT-4, Llama, BLOOM, Falcon, ...
 - Decoders shouldn't be used for embedding.
![Decoders](/GenerationsByDecoders.png)

## Encoders-Decoders
 - Encoders-decoders - encodes a sequence of words and use the encoding + to output a next word.
 - Examples includes - T5, UL2, BART, ...
 - The below image is for a translator.
![Encoders&Decoders](/Encoders&Decoders.png)
 - Not all models are suitable for all tasks. The below image gives an overview on this :
![ArchitectureAtGlance](/ArchitectureAtGlance.png)