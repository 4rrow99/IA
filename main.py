from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_agents import create_tool_calling_agent
from langchain_agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


load_dotenv()

###### For research

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    soucers: list[str]
    tools_used: list[str]


llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

parser= PydanticOutputParser(pydantic_object=ResearchResponse)
######Theorical, need the keys

response = llm.invoke("who win the ucl 2002")
print(response)

#### do it in env)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            you are a research assistant that will help generate a research paper. 
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_isntructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(

    llm=llm,
    prompt=prompt,
    tools=tools 
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("what can i help you")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e: 
    print("Error parsion response", e, "Raw Response - ", raw_response)
