from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/faiss_db"

# Custom PromptTemplate
custom_prompt = """
Use the following prompt to retrieve information from the database:
Question: {question}
Context: {context}

Answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt, input_variables=["question", "context"])
    return prompt

def load_llm():
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin", 
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.5,)
    return llm

def retrieve_answer(llm, prompt, db):
    qa = RetrievalQA()
    return qa

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                        model_kwargs = {
                                             'device': 'cpu',
                                        })
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieve_answer(llm, qa_prompt, db)
    return qa

def final_result(query):
    result = qa_bot()
    response = result({'question': query})
    return response


# chainlit
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message("Bot is ready to answer your questions!")
    await chain.send(msg)
    msg.content = "Welcome to the Med QA Bot! Ask me anything about medicine."
    await msg.update()
    cl.set_context("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.use_context("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_token=["FINAL", "ANSWER"]
    ),
    cb.answer_reached=True
    res = await chain.acall(message.content, callback=cb)
    answer = res["final_answer"]
    source = res["source_documents"]

    if source:
        answer += f"\n\nSource: {source[0].title}"
    else:
        answer += "\n\nSource: No source found"

    await cl.Message(content = answer).send()