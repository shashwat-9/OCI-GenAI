# Overview
 - OCI Generative AI Service is provided by oracle.
 - Topic to be covered in this module :
    1. Pretrained Foundational Models
    2. Prompt Engineering and LLM customization
    3. Fine-tuning and Inference
    4. Dedicated AI Clusters
    5. Generative AI Security Architecture

# Introduction to OCI Generative AI Service
 - Fully managed service that provides a set of customizable Large Language Models(LLMs) available via a single API to
build generative AI applications
 - Choice of Models: high performance pretrained models from Meta and Cohere
 - Supports Flexible Fine-tuning : create custom models by fine-tuning foundational models with our own data set
 - Dedicated AI Clusters: GPU based compute resources that host your fine-tuning and interface workloads

## How does the OCI Generative AI service work?
 - We give input in Natural Language, the OCI Gen AI processes the language and do the requirements and produces a result
 - Built to understand, generate, and process human language at a massive scale
 - Use-cases: Text Generation, Summarization, Data Extraction, Classification, Conversation

### Pretrained Foundational Models
 - OCI Gen AI provides several models like Generation Model
 - The models available under this section are: command and command-light by cohere, and Llama 2-70B-chat by Meta
 - The basic idea behind these model is Text Generation and is a Instruction-following model, that is we can provide our
own data set to train them called instruction tuning.
 - The second category of model available is Text summarization model. Summarize text with your instructed format, length
, and tone. The available model Command by cohere.
 - The third category is Embedding models. Embedding is used to convert text to vector embeddings, used for semantic 
searches, where the search function focuses on the meaning of the text that it is searching through rather than finding 
results based on keywords. Embedding makes it easier for the machine to understand the pieces of text.
 - The embedding models are MultiLingual model. That is, we can use multiple languages across multiple queries, e.g. 
french query on english document. These models generally supports 100+ languages.
![PreTrainedModels](/PreTrainedModels.png)

### Fine Tuning
 - A key capability of OCI Gen AI is to fine tune these pre-trained foundational models.
 - It basically means optimizing a pretrained foundational model on a smaller domain specific dataset.
   1. Improve Model performance on a specific task
   2. Improve Model Efficiency
![CustomModels](/CustomModels.png)
 - Use when a pretrained model doesn't perform your task well or you want to teach it something new.
 - OCI Generative AI uses the T-few fine-tuning to enable fast and efficient customizations.
 - T-few is a fine-tuning technique, where we update only a portion of model's weight. Doing so gives us better accuracy
and lower cost.

### Dedicated AI Clusters
 - Dedicated AI cluster are GPU based compute resources that host the customer's fine-tuning and inference workloads.
 - Gen AI service establishes a dedicated AI cluster, which includes dedicated GPUs and an exclusive RDMA cluster network
for connecting the GPUs.
 - The GPUs allocated for a customer's generative AI task are isolated from other GPUs.
![GenAIClusters](/GenAIClusters.png)

### Demo: Generative AI service Walkthrough
 - Open the OCI account-> click on the navigation menu -> Analytics & AI -> AI Services -> Generative AI
 - This will open the Gen AI console
 - On the left side, we can see the compartment listed
 - On the overview console, there are few tabs :
   1. Playground -> Here we can test the pre-trained models
   2. Dedicate AI clusters -> sign up dedicated hardware for AI training
   3. Custom models -> Create custom model by fine-tuning the base models.
   4. Endpoints -> Here we can get endpoints for our custom generative AI models

#### 1. PlayGround
 - Here we get to choose from the text models, various parameters like temperature and then we input our text to find
the output.
 - We also get the code for connecting our application to this Gen AI instance in java/python.

### Generation Models

#### Tokens
 - Language models understands "tokens" rather than characters.
 - One token can be a part of word, an entire word, or punctuation
   1. A common word such as "apple" is a token
   2. A word such as friendship is made up of two tokens - "friend" and "ship"
 - Number of Tokens/Word depend on the complexity of the text
   1. Simple text: 1 token/word (Avg.)
   2. Complex text(less common words): 2-3 tokens/word(Avg.)

#### Pretrained Generation Models in Generative AI
 - Context windows - No of tokens a model is capable of processing at one time. It is the sum of input and output token 
from that model.
 1. Command Model from cohere
    - Highly performant, instruction-following conversational model
    - Model Parameters: 52B, context window: 4096 tokens
    - Use cases: text generation, chat, text summarization

 2. Command-Light from cohere
    - Smaller, faster version of Command, but almost as capable
    - Model Parameters: 6B, context window: 4096 tokens
    - Use when speed and cost are important (give clear instructions for best results)

 3. Llama-2-70b-chat
    - Highly performant, open-source model optimized for dialogue use cases
    - Model parameters: 70B, context window: 4096 tokens
    - Use cases: chat, text generation

#### Generation Model Parameters
 - Maximum output tokens -> max number of tokens model generates per response. In case of OCI, the limit is 4000 tokens
 - Temperature -> Determines how creative the model should be; close second to prompt engineering in controlling the
output of generation models.
 - Top p, Top k -> Two additional ways to pick the output tokens besides temperature
 - Presence/Frequency Penalty -> Assigns a penalty when a token appears frequently and produces less repetitive text
 - Show Likelihoods -> Determines how likely it would be for a token to follow the current generated token

#### Temperature
 - Temperature is a (hyper) parameter that controls the randomness of the LLM output.
```
    The sky is ______
    
    word        blue    the limit   red     tarnished   water
    probability 0.45    0.25        0.20    0.01        0.02
```
 - Temperature of 0 makes the model deterministic (limits the model to use the word with the highest probability)
 - When temperature is increased, the distribution is flattened over all words.
 - With increased temperature, model uses words with lower probabilities.

#### Top k
 - Top k tells the model to pick the next token from the top 'k' tokens in its list, sorted by probability.

#### Top p
 - Top p is similar to Top k but picks from the top tokens based on the sum of their probabilities.
 - Basically, it picks up from the set of words whose sum total of probability is < top k

#### Stop Sequences
 - A stop sequence is a string that tells the model to stop generating more content
 - It is a way ti control your model output
 - If a period(.) is used as a stop sequence, the model stops generating text once it reaches the end of the first sentence,
even if the number of tokens limit is much higher.

#### Frequency and presence penalties
 - These are useful if you want to get rid of repetitions in your outputs.
 - Frequency penalty penalizes tokens that have already appeared in the preceding text (including the prompt), and scales
based on how many times that token has appeared.
 - So a token that has already appeared 10 times gets a higher penalty (which reduces its probability of appearing) than
a token that has appeared only once
 - Presence penalty applies the penalty regardless of frequency. As long as the token has appeared once before, it will
get penalized.

#### Show likelihoods
 - Every time a new token is to be generated, a number between -15 and 0 is assigned to all tokens.
 - Tokens with higher numbers are more likely to follow the current token
 - These number can be viewed after the generation of output is done

### Demo: Generation Models
 - Three different scenarios are to be discussed :
    1. Text Generation -> Used to generate text
       e.g. : Generate a tag line for an ice-cream shop
    2. Data Extraction -> Extract particulars out of the given report
        e.g. : Extract company name, theme, and discussion points in the following reports. ...report...
    3. Text Classification -> A text is classified as some value, and the next text is given asking for the classification
        e.g. ->  statement : s1, classification: c1;
                 statement : s2, classification: c2; 
             ... statement : sn, classification: cn
                statement : s(n+1), classification : ?

### Demo: OCI Generative AI Service Inference API
 - Generative AI service Inference API is the endpoint through which we can call the custom or out of the box models via
a code than the console.
 - The code that can be used for this purpose could be copied through the view code tab in the top right corner in the 
console. It is available in both java & python.
 - The script uses a config file along with a few other parameters like compartment id etc. , to call the api for the 
desired output.  

 - The config file looks like the following :
```
    [Profile1]
    user=
    fingerprint=
    tenancy=
    region=
    key_file=
    
    [profile2]
    ...
```
 - The key file is a .pem file used for authentication.
 - A new .pem file can be generated by going to the user profile -> Add API key -> download the new file -> It will give
the new parameters to be filled in the config file(the above format).

### Summarization Model
 - Generates a succinct version of the original text that relays the most important information
 - Same as one of the pretrained text generation models, but with parameters that you can specify for text summarization
 - Use cases include, but not limited to: News articles, blogs, chat transcript, scientific articles, meeting notes, and
any text that you should like to see a summary of.

#### Summarization Model Parameters
 - Temperature -> Determines how creative the model should be; Default temperature is 1 and the maximum temperature is 5
 - Length -> Approximate length of the summary. Choose from short, medium, and Long
 - Format -> Whether to display the summary in a free-form paragraph or in bullet points
 - Extractiveness -> How much to reuse the input in the summary. Summaries with high extractiveness lean towards reusing
sentences verbatim, whereas summaries with low extractiveness tend to paraphrase.

#### Embedding Models
 - Embeddings are numerical representations of a piece of text converted to number sequences
 - A piece of text could be a word, phrase, sentence, paragraph or one or more paragraph
 - Embeddings make it easy for computers to understand the relationships between pieces of text

##### Word Embeddings
 - Word Embeddings capture properties of the word
 - Vectors actually are a sequence of numbers where each number represent some properties in n-dimension
![WordEmbeddings](/WordEmbeddings.png)

##### Semantic Similarity
 - Embeddings are basically conversion of words into vector of numbers.
 - So these vector representations can be leveraged to compute similarity.
 - cosine and dot products similarly can be used to compute numeric similarity
 - Embeddings that are numerically similar are also semantically similar
 - E.g. embedding vector of "Puppy" will be more similar to "Dog" than that of "Lion"
![SemanticSimilarity](/SemanticSimilarity.png)

##### Sentence Embeddings
 - A sentence embedding associates every sentence with a vector of numbers
 - Similar sentences are assigned to similar vectors, different sentences are assigned to different vectors
![SentenceEmbedding](/SentenceEmbedding.png)

##### Embedding use case
 - Vector databases are capable of automating the cosine similarity and doing the nearest match searches for that database.
 - When a query is fed into 
![EmbeddingUseCases](/EmbeddingUseCases.png)

##### Embedding Models in Generative AI
![EmbeddingModelsInGenAI](/EmbeddingModelsinGenAI.png)
 - Cohere.embed-english converts English text into vector embeddings
 - Cohere.embed-english-light is the smaller and faster version of embed-english
 - Cohere.embed-multilingual is the state-of-the-art multilingual embedding model that can convert text in over 100
languages into vector embeddings. It can also enable searching within a language, so that you can search a french query
on french document, or french query on english document.
 - Use cases: semantic search, Text Classification, Text Clustering

 - Recently embed-english-v3.0 is launched with many features including:
    1. English and Multilingual
    2. Model creates a 1024-dimensional vector for each embedding
    3. Max 512 tokens per embedding
    4. The key improvement in v3.0 is the ability to evaluate how well a query matches a document and assess the overall
quality of document. This means ranking high quality documents at the top, which means dealing with noisy datasets.

    - There's also a light mode of the above available with the following features :
      1. Smaller, faster version; English and Multilingual
      2. Model creates a 384-dimensional vector for each embedding
      3. Max 512 tokens per embedding

    - Then there's this old(previous) embedding model :
        1. Previous generation models, English
        2. Model creates a 1024 dimensional vector for each embedding
        3. Max 512 tokens per embedding

![EmbeddingModels](/EmbeddingModels.png)
 - A maximum of 96 inputs are allowed for each run, and each input must have less than 512 tokens

### Demo: Summarization and Embedding Models
 - We can select the Summarization model from the drop-down and then adjust parameters like length, extractiveness etc.
 - About the embedding model, when we enter sentences to embed, a 2-D equivalent plot is created in the output box.
 - In that output, we can see that similar sentences points are located near to each other.
 - If we run the code from the view code section, that calls the api to embed the sentences, we will get the vectors in
return.

### Prompt Engineering

 - Prompt -> The input or initial text provided to the model
 - Prompt Engineering -> The process of iteratively refining a prompt for the purpose of eliciting a particular style of
response

#### LLMs as next word predictors
 - Text prompts are how users interacts with large language models
 - LLM models attempt to produce the next series of words that are most likely to follow from previous text.

#### Aligning LLMs to follow instructions
 - Completion LLMs are trained to predict the next word on a large dataset of internet text, rather than to safely
perform the language task that the user wants.
 - Cannot give instruction or ask question to a completion LLM
 - Instead, need to formulate your input as a prompt whose natural continuation is your desired output
 - But this is not practical and how LLMs actually works. It is further fine-tuned with various new researches to make it
work as today.
 - Reinforcement Learning from Human Feedback is used to fine-tune LLMs to follow a broad class of written instructions
 - Llama-2 base model was trained with 2 Trillions tokens, while Llama-2 chat, which is different from the base model is
additionally trained on 28 thousands prompt-response pair.
 - Most of the LLMs today can follow instructions because they're fine-tuned to do so

#### In-context Learning and Few-shot Prompting
 - In-Context Learning -> conditioning(prompting) an LLM with instructions and or demonstrations of the task it is meant
to complete. Here, no parameter is changing, or no learning is made, rather the LLM is fed what to complete and a
demonstration of the task.
 - K-shot prompting -> provides k examples and then the requirement.

#### Prompt Formats
 - Large Language Models are trained on a specific prompt format. If you format prompt in a different way, you may get
odd/inferior results.
 - Llama2 Prompt Formatting:
```
    <<s>>       -- Beginning of the entire sequence 
    [INST]      -- Beginning of instructions
       <<SYS>>
       {{system_prompt}}    --System Prompt to set model context
       <</SYS>>
       {{user_message}}
    [/INST]     -- End of Instructions
```
 - If we want an optimal result, we should follow the above format Llama-2

#### Advanced Prompting Strategies
 - Chain-of-Thought -> provide examples in a prompt is to show responses that include a reasoning step
 - Zero Shot Chain-of-thought -> Apply chain of thought prompting without providing examples

### Demo: Prompt Engineering with OCI Generative AI
 - In Chain-of-thought prompting technique, we provide the example reasoning to solve the problem. And expect teh same in
output.
 - In Zero-shot Chain-of-Thought Prompting, we add something like "Let's think step by step" at the end, rather than the
whole logic.

 - The Oracle Generative AI has a single inference API to call any Foundational Models
 - If we don't use the Llama2 Prompt formatting while prompting to this particular model, we will get suboptimal results.
Like maybe some other answers along with what is required.
 - In the sys tag, we give the characteristics of the system, like respectful, helpful etc.

### Customize LLMs with your data

#### Training LLMs from scratch with my data?
 - There are various issues in doing so :
   1. Cost - Expensive
   2. Data - A lot of data is needed
   3. Expertise - Pretraining a model is hard, and requires through understanding of the process.

 - There are other ways of doing so :
    1. In-context Learning/Few shot prompting -> 
       - User provides demonstrations in the prompt to teach the model how to perform certain tasks.
       - Popular techniques include Chain of Thought Prompting
       - Main limitation: Model Context Length

    2. Fine-tuning a pretrained model
       - Optimize a model on a smaller domain-specific dataset
       - Recommended when a pretrained model doesn't perform your task well or when you want to teach it something new
       - Adapt to specific style and tone, and learn human preference

![FineTuningTechnique](/FineTuningTechnique.png)

##### Fine-tuning Benefits
 - Improve Model Performance on specific tasks
   1. More effective mechanism of improving model performance than Prompt Engineering
   2. By customizing the model to domain-specific data, it can better understand and generate contextually relevant responses.

 - Improve Model Efficiency
    1. Reduce the number of tokens needed for your model to perform well on your tasks.
    2. Condense the expertise of a large model into a smaller, more efficient model.

##### Retrieval Augmented Generation (RAG)
 - Language model is able to query enterprise knowledge bases(databases, wikis, vector databases, etc.) to provide grounded
responses. Grounded means, the text is in some document.
 - RAGs do not require custom models

![CustomLLMsCategorization](/CustomLLMsCategorization.png)

##### Customize LLMs with your data
 - Prompt Engineering is the easiest to start with; test and learn quickly.
 - If you need more context, then use Retrieval Augmented Generation(RAG).
 - If you need more instruction following, then use fine-tuning

![CustomizeLLMswithData](/CustomizeLLMswithData.png)

### Fine-Tuning and Inference in OCI Generative AI
 - A model is fine-tuned by taking a pretrained foundational model and providing additional training using custom data.
 - In Machine Learning, Inference refers to the process of using a trained ML model to make predictions or decisions 
based on new input data.
 - With Language models, inference refers to the model receiving new text as input and generating output text based on 
what it has learned during training and fine-tuning.
![FineTuning&Inferencing](/FineTuning&Inferencing.png)

#### Fine-tuning workflow in OCI Generative AI
 - Custom Model: A model that you can create by using a Pretrained Model as a base and using your own dataset to 
fine-tune that model.
```
    The flow is as :
    1. Create a Dedicated AI Cluster
    2. Gather Training Data
    3. KickStart Fine-tuning
    4. Fine-tuned (custom) Model gets created.
```

#### Inference workflow in OCI Generative AI
 - Model Endpoint -> A designated point on a dedicated AI cluster where a large language model can accept user requests
and send back responses such as the model's generated text
```
    1. Create Dedicated AI Cluster(Hosting)
    2. Create Endpoint
    3. Serve Model
```

#### Dedicated AI Clusters
 - Effectively a single-tenant deployment where the GPUs in the cluster only host your custom models.
 - Since the model endpoint isn't shared with other customers, the model throughput is consistent
 - The minimum cluster size is easier to estimate based on the expected throughput
 - Cluster Type
    1. Fine-tuning -> used for training a pretrained foundational model
    2. Hosting -> Used for hosting a custom model endpoint for inference

#### T-few Fine-Tuning 
 - Traditionally, vanilla fine-tuning involves updating the weights of all (most) the layers in the model,
requiring longer training time and higher serving (inference) costs.
 - T-few fine-tuning selectively updates only a fraction of the model's weights.
    1. T-few fine-tuning is an additive Few-Shot Parameter Efficient Fine-Tuning (PEFT) technique that inserts additional
layers, comprising ~0.01% of the baseline model's size.
    2. The weight updates are localized to the T-few layers during the fine-tuning process
    3. Isolating the weight updates to these T-few layers significantly reduces the overall training time and cost compared
to updating all layers.

 - T-few fine-tuning process begins by utilizing the initial weights of the base model and an annotated training dataset
 - Annotated data comprises of input-output pairs employed in supervised training
 - Supplementary set of model weights is generated (~0.01% of the baseline model's size)
 - Updates to the weights are confined to a specific group of transformer layers.(T-few transformer layers), saving
substantial training time and cost.
![T_FewFineTuningProcess](/T_FewFineTuningProcess.png)

#### Reducing Inference Costs
 - Inference is computationally expensive
 - Each hosting cluster can host one base Model endpoint and up to N fine-tuned custom model endpoints serving request 
concurrently.
 - This approach of models sharing the same GPU resources reduces teh expenses associated with inference
 - Endpoints can be deactivated to stop serving requests and reactivated later.

![InferencingCost](/InferencingCost.png)

#### Inference serving with minimal overhead
 - GPU memory is limited, so switching between models can incur significant overhead due to reloading the full GPU memory
 - These models share the majority of weights, with only slight variations; can be efficiently deployed on the same GPUs
in a dedicated AI cluster.
 - This architecture results in minimal overhead when switching between models derived from the same base model
![InferenceServingWithMinimalOverhead](/InferenceServingWithMinimalOverhead.png)
