from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.prompts import MessagesPlaceholder
from openai import OpenAI
from langchain.tools import StructuredTool
from langchain.pydantic_v1 import Field, create_model
from inspect import signature
from typing import Callable
import os
import textwrap
import re

os.environ["OPENAI_API_KEY"] = "ADD-KEY-HERE"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_completion(messages, model="gpt-4o-mini", temperature=0.1):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature = temperature

    )

    return chat_completion.choices[0].message.content


class Chatbot:
    def __init__(self):
        self.reset()
        self.prompts = 0

    def get_completion(self, prompt: str) -> str:
        self.history.append({"role": "user", "content" : prompt})
        completion = get_completion(self.history)
        self.history.append({"role": "assistant", "content" : completion})
        return completion
    
    def reset(self):
        self.history = []


# credit: https://github.com/langchain-ai/langchain/discussions/9404
def create_tool(callable:Callable):
    method = callable
    args = {k:v for k,v in method.__annotations__.items() if k not in  ["self", "return"]}
    name = method.__name__
    doc = method.__doc__
    func_desc = doc[doc.find("<desc>") + len("<desc>"):doc.find("</desc>")]
    arg_desc = dict()
    for arg in args.keys():
        desc = doc[doc.find(f"{arg}: ")+len(f"{arg}: "):]
        desc = desc[:desc.find("\n")]
        arg_desc[arg] = desc
    arg_fields = dict()
    for k,v in args.items():
        arg_fields[k] = (v, Field(description=arg_desc[k]))

    Model = create_model('Model', **arg_fields)

    tool = StructuredTool.from_function(
        func=method,
        name=name,
        description=func_desc,
        args_schema=Model,
        return_direct=False,
    )
    return tool

class AgentSmith:
    def __init__(self, tools = []):
        self.custom_tools = {}
        for tool in tools:
            self.custom_tools[tool.func.__name__] = tool
        self.bot = Chatbot()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a powerful assistant who can overcome problems by creating tools to solve them programmatically. Be very persistent in solving problems. IMPORTANT: Only create one tool at a time.""",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        ) 
        
        self.tools = [create_tool(self.create_tool), create_tool(self.ask_gpt), create_tool(self.consult_agent)] + tools
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.chat_history = []
        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | self.prompt
            | self.llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.tools, verbose=True)


    def run(self, prompt):
        conversation = list(self.agent_executor.stream({"input": prompt}))
        print(conversation)
    
    def consult_agent(self, prompt: str) -> str:
        """Consult the GPT 4o agent
        """

        agent = AgentSmith(self.custom_tools)
        conversation = list(agent.agent_executor.stream({"input": prompt}))

        header = "This was the response from the GPT 4o agent. Intelligently use its response to complete your task.\nRESPONSE:\n"
        return header + conversation[-1]["output"]

    def ask_gpt(self, prompt: str) -> str:
        """Send a prompt to GPT 4o Mini.
        """
        header = "This was the response from GPT 4o. Intelligently use its response to complete your task.\nRESPONSE:\n"
        return header + self.bot.get_completion(prompt)
    
    @staticmethod
    def get_def(text):
        pattern = re.compile(rf"^def.*$", re.MULTILINE)
        match = pattern.search(text)
        if match:
            return match.group(0)
        else:
            return None
    
    @tool
    def reset(self) -> str:
        """If you encounter errors, run this tool to get a clean slate.
        """
        self.custom_tools = []
        self.tools = [create_tool(self.create_tool), create_tool(self.ask_gpt), self.reset]
    

    def create_tool(self, code: str) -> str:
        """Enter syntactically-correct python code for a function to create a new tool. IMPORTANT: Only create one tool at a time. When writing code, you must satisfy the following requirements to be able to use the tool:
            1. Inclusion of a docstring after the function signature.
            2. Specification of input and return types in the function signature. For example: 'def test(input: str) -> str'. 
            3. The input and output types MUST be primitives.
            4. Implement the tool completely. 
        """
        code=code.replace('\\n', '\n')
        def_line = self.get_def(code)
        function_name = def_line.split('(')[0].split(' ')[1]
        # print("func name", function_name)
        # if function_name in globals():
        #     return "You have already created that tool."

        try:
            exec(code, globals())
        except Exception as e:
            return "The code you provided has an error: \n" +  str(e)
        
        func = globals()[function_name]
        input_dict = "{" + ", ".join(list(map(lambda x: f""""{x}": {x}""", func.__code__.co_varnames[:func.__code__.co_argcount]))) + "}"
        code = textwrap.indent(re.sub(r'^(import|from).*\n?', '', code, flags=re.MULTILINE), 4 * ' ')
        
#         wrapped_code = f'''
# def {function_name}{signature(func)}:
#     """{func.__doc__}"""
#     @tool
# {code}
#     try:
#         print("input dict", {input_dict})
#         return {function_name}.invoke({input_dict})
#     except Exception as e:
#         return "The tool you executed encountered an error. Think about why you got the error, then write new tools to fix it.\\nError:\\n" + str(e)
# ''' 
        
        # below is correct code for wrapper
        wrapped_code = f'''
def {function_name}{signature(func)}:
    """{func.__doc__}"""
{code}
    try:
        return {function_name}({', '.join(func.__code__.co_varnames[:func.__code__.co_argcount])})
    except Exception as e:
        return "The tool you executed encountered an error. Think about why you got the error, then write new tools to fix it.\\nError:\\n" + str(e)
''' 
        print(wrapped_code)
        try:

            exec(wrapped_code, globals())
        except:
            return "Error: Input and output types must be primitives and a docstring must be provided."
        

        self.custom_tools[function_name] = create_tool(globals()[function_name])

        new_agent_executor = AgentSmith(list(self.custom_tools.values())).agent_executor
        self.agent_executor.tools = new_agent_executor.tools
        self.agent_executor.agent = new_agent_executor.agent

        return "Tool created successfully"

agent = AgentSmith()

while True:
    with open('prompt.txt', 'r') as prompt:
        text = prompt.read()
        print(text)
    agent.run(text)