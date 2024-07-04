## Architechture

- The project starts with collecting the data from a pdf and then converting it into chunks of text. Because full pdf is not readable by the model since there is a token limit of 4096. So, we have to convert the pdf into chunks of text.
- Then we convert the chunks of data into vector embeddings and apply semantic indexing on it.
- Semantic indexing is used to categorize the data into different catatogies.
- Then we store the data as our knowledge base into a vector database, which will be pinecone in our case.

-> The user will then ask a question will be converted into a query embedding and then we will search the knowledge base for the most relevant answer to the query. The result will be a ranked result of the answers to the query. From which our model llama-2 will select the best answer and return it to the user.

## Tech Stack

- Python
- Flask (Frontend)
- Pinecone (Vector Database, remotely hosted)
- Llama-2 (LLM Model)
- Langchain (Generative AI Framework)

## How to run the project

```bash
conda create -n medicalChatbot python=3.8 -y
conda activate medicalChatbot
pip install -r requirements.txt
```
