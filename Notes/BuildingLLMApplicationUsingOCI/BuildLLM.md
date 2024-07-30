# Introduction

 - Build a Conversational Chatbot
    1. LangChain Prompts and Models
    2. Incorporate Memory
    3. Implemented Retrieval Augmented Generation with LangChain
    4. Trace LLM calls and Evaluate with LangSmith
    5. Deploy Chatbot on OCI

## Chatbot Introduction

 - Recap
    1. LLMs
    2. OCI Generative AI Service
    3. Vector Databases
    4. Embeddings
    5. Semantics Search
    6. RAG

 - We will create this chatbot to answer questions about OCI Certification courses.
   1. OCI Generative AI Service -> We will use OCI Generative AI as LLM to answer our queries
   2. LangChain -> We will use LangChain framework to build our Chatbot application.
   3. Chatbot will use custom relevant documents to answer questions.

## Chatbot Architecture & Basic Components

 - The way LLMs work is that a question/prompt is provided to the LLM, and as an add-on we can retrieve documents from storage
and can also add previous chat history from the memory and then prompt the LLM. The LLM then generate the answer, which
can get stored in the chat memory.
![ChatbotArchitecture](/ChatbotArchitecture.png)

#### OCI Generative AI and LangChain Integration
 - Using the OCI Generative AI service you can access pretrained models or create and host your own fine-tuned custom models
based on your own data on dedicated AI clusters.
 - Open source framework like langchain offers multitude of components to build LLM based application including with OCI
generative service, vector databases, document loaders and many others. 
 - langchain_community provides a wrapper class for using OCI Generative AI service as an LLM in LangChain Applications.
   ```
      langchain_community.llms.OCIGenAI
   ```

#### LangChain Components
 - LangChain is a framework for developing applications powered by language models.
 - It offers a multitude of components that help us build LLM-powered application.
 - A few components that are used to build our Chatbot:
   1. LLMs
   2. Prompts
   3. Memory
   4. Chains
   5. Vector Stores
   6. Document Loaders
![LangChainApplication](/LangChainApplication.png)

## Models, Prompts and Chains
 - The heart of the LLM application is the Large Language Model itself.
 - There are two main types of models that LangChain integrates with :
   1. LLM -> LLMs in LangChain refers to pure text completion models. They take a string prompt as input and output a 
string completion
   2. Chat Models -> Chat models are often backed by LLMs but are tuned specifically for having converstaions.
They take a list of chat messages as input and return and AI message as output.
![LangChainModels](/LangChainModels.png)

#### LangChain Prompt Template
 - Langchain have pre-built classes that we will use to create prompts.
 - Prompts templates are predefined recipes for generating prompts for language model.
 - Typically, language models expect the prompt to either be a string or else a list of chat messages.
 
1. String Prompt Template -> The template supports any number of variables, including no variables
2. Chat Prompt Template -> Using this we can input a list of message to chat model. Each chat message is associated with
content, and an additional parameter called role.

![LangChainPromptTemplate](/LangChainPromptTemplate.png)

#### LangChain Chains
 - LangChain provides framework for creating chains of component including LLMs and other type of components
 - We can create LangChains by two means:
   1. Using LCEL -> Create chains declaratively using LCEL. LangChain Expression Language, or LCEL, is a declarative way
to easily compose chains together.
   2. Legacy -> Create chains using Python classes like LLM Chain and others.
![SettingUpDevEnv](/SettingUpDevEnv.png)

 - In pycharm dev IDE, we can install python packages by clicking on python packages from the bottom bar.
 - Packages required in this project are : OCI, Oracle IDEs, LangChain, LangSmith, Chroma DB, Wise, Pydantic, Streamlit,
and PyPDF

#### Demo: Prompts, Chains, and LLMs
 - LLMChain objects can be used to create chains by passing the llm,prompt and stroutputparser() in the constructor.
 - We can use | operator in to connect the prompt, llm and thus to create chain
 - The relevant code can be found here [LinkTofolder](./code)