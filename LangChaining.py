#Lars V.

import re
import janus_swi as janus

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field

class PrologOutput(BaseModel):
    knowledge_base: str = Field(description="Prolog facts and rules")
    query: str = Field(description="The Prolog query")
    #answer: str = Field(description="The answer from the prolog_query function")

# utility
# for a predicate is string form, True if there is at least one variable
# (= uppercase) in argumentlist
def has_vars(predstring):
      regex = r"[a-z][a-zA-Z0-9_]*\(([A-Z].*|.*,[]*[A-Z].*)\)"
      match = re.match(regex, predstring)
      if match:
            return True
      return False


def prolog_query(knowledge_base: str, query: str) -> dict:
    """
    Execute a Prolog program and query using SWI-Prolog through Janus-SWI.
    """
    janus.consult("knowledge base", data = knowledge_base)
    thequery = query.lstrip('?- ').rstrip('.')

    result = janus.query_once(thequery)
    return (thequery, result)

def report_answer(thequery, result):
    print(f"Here is the answer to query {thequery}:\n")

    if result['truth'] == True:
        del result['truth']
        if has_vars(thequery):
                print(f"TRUE with {result}")
        else:
            print("TRUE")
    else:
        print("FALSE")
    print()
    return

#tools = [prolog_query]

print("Enter your text. Start be describing the facts and rules of your knowledge domain\n\
      as you know them. Type END on a line by itself to finish:")

input_knowledge = ""
while True:
    line = input()
    if line == "END":
        break
    input_knowledge += line

input_question = input("\nNow ask your question: ")

input_instructions = """
    Translate the following into Prolog.
    Return ONLY valid Prolog code.
    Do not include explanations.
    Do not include Markdown.
    Do not include check marks or emojis.
    Output only facts, rules, and queries.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are a logic program assisstant.
        You translate language into logic programs following the instructions.
        
        Knowledge base : {knowledge}

        Question: {question}

        Instructions: {instructions}
		
        Translate the knowledge base into Prolog facts and rules. 
        Also translate the question into a Prolog query.
        """),

        ("human","""
        Knowledge: {knowledge}

        Question:{question}
        """)   
	]
)

llm = ChatOllama(model="qwen3", temperature=0)
structured_llm = llm.with_structured_output(PrologOutput)
#structured_llm_with_tools = structured_llm.bind_tools(tools)

chain = prompt | structured_llm

result = chain.invoke({"knowledge" : input_knowledge,
                        "question" : input_question,
                        "instructions" : input_instructions
})

print()
print(result.knowledge_base)
print()
print(result.query)
print()

(plain_query, answer) = prolog_query(result.knowledge_base, result.query)
report_answer(plain_query, answer)
print()






