from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="tinyllama")

template = """

You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews:{reviews}

Here is the question to answer:{question}

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    question = input("Enter a question(q to quit): ")
    if question == "q":
        break
    reviews = retriever.invoke(question)
    res = chain.invoke({"reviews": [], "question":question})
    print(res)