# Retrieval-Augmented-Generation-On-Azure

## Introduction

Knowledge bases are widely used in enterprises and can contain an extensive number of documents across various categories. Retrieving relevant content based on user queries is a challenging task. Traditionally, methods like Page Rank have been employed to accurately retrieve information at the document level. However, users still need to manually search within the document to find the specific and relevant information they need. With the recent advancements in Foundation Models, such as the one developed by OpenAI, offer a solution through the use of "Retrieval Augmented Generation" techniques and encoding information like "Embeddings." These methods aid in finding the relevant information and then to answer or summarize the content to present to the user in a concise and succinct manner.

Retrieval augmented generation (RAG) is an innovative approach that combines the power of retrieval-based Knowledge bases, such as Azure Cognitive Search, and generative Large Language Models (LLMs), such as Azure OpenAI ChatGPT, to enhance the quality and relevance of generated outputs. This technique involves integrating a retrieval component into a generative model, enabling the retrieval of contextual and domain-specific information from the knowledge base. By incorporating this contextual knowledge alongside the original input, the model can generate desired outputs, such as summaries, information extraction, or question answering. In essence, the utilization of RAG with LLMs allows you to generate domain-specific text outputs by incorporating specific external data as part of the context provided to the LLMs.

RAG aims to overcome limitations found in purely generative models, including issues of factual accuracy, relevance, and coherence, often seen in the form of "hallucinations". By integrating retrieval into the generative process, RAG seeks to mitigate these challenges. The incorporation of retrieved information serves to "ground" the large language models (LLMs), ensuring that the generated content better aligns with the intended context, enhances factual correctness, and produces more coherent and meaningful outputs.

## Description

This [end-to-end notebook](end-to-end-RAG.ipynb) covers all the concepts of RAG:
  * **Process:** Use Azure Form Recognizer to convert unstructured data from raw documents (stored in Azure Blob Storage container) to a more structured JSON format that can be indexed in Azure Cognitive Search.
  * **Retrieve:** Use Azure Cognitive Search to create an index of documents and search/retrieve relevant contextual information based on the user questions.
  * **Generate:** Use Azure OpenAI service to generate answers to user questions by using the retrieved information in the prompts.
    
**Note:** Update the credentials of your Azure resources in the ``.env`` file


### **Tutorials:**
* [RAG for structured data on Azure](tutorial1-RAG_structured_with_cognitive_search.ipynb)
* [RAG for unstructured data on Azure](tutorial2-RAG_unstructured_with_cognitive_search.ipynb)




## Some Considerations:

* **Evaluation challenges:** Evaluating the performance of RAG poses challenges, as traditional metrics may not fully capture the improvements achieved through retrieval. Developing task-specific evaluation metrics or conducting human evaluations can provide more accurate assessments of the quality and effectiveness of the approach.
* **Ethical considerations:** While RAG provides powerful capabilities, it also introduces ethical considerations. The retrieval component should be carefully designed and evaluated to avoid biased or harmful information retrieval. Additionally, the generated content should be monitored and controlled to ensure it aligns with ethical guidelines and does not propagate misinformation or harmful biases.

