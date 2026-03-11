#
# 用更底层的方式实现了一个基于 ReAct 框架的 Agent。
# 如何编写一个 ReAct 提示词，让大模型通过思考（Thought）、行动（Action）、观察（Observation）来完成更多的工作。
# Agent 的执行过程本质上就是一个循环，由大模型引导着一次次地做着各种动作，以达成我们的目标。
#

import re
import sys
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from dotenv import load_dotenv

load_dotenv()
print("loaded .env file")


client = OpenAI()


class Agent:
    def __init__(self, system="") -> None:
        self.system = system
        self.messages: list[ChatCompletionMessageParam] = (
            [{"role": "system", "content": system}] if system else []
        )

    def _execute(self):
        completion = client.chat.completions.create(
            model="deepseek-chat", messages=self.messages, temperature=0
        )
        return completion.choices[0].message.content or ""

    def invoke(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self._execute()
        self.messages.append({"role": "assistant", "content": result})
        return result


prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

ask_fruit_unit_price:
e.g. ask_fruit_unit_price: apple
Asks the user for the price of a fruit

Example session:

Question: What is the unit price of apple?
Thought: I need to ask the user for the price of an apple to provide the unit price. 
Action: ask_fruit_unit_price: apple
PAUSE

You will be called again with this:

Observation: Apple unit price is 10/kg

You then output:

Answer: The unit price of apple is 10 per kg.
""".strip()


def calculate(what):
    return eval(what)


def ask_fruit_unit_price(fruit):
    if fruit.casefold() == "apple":
        return "Apple unit price is 10/kg"
    elif fruit.casefold() == "banana":
        return "Banana unit price is 6/kg"
    else:
        return "{} unit price is 20/kg".format(fruit)


action_re = re.compile(r"^Action: (\w+): (.*)$")

known_actions = {"calculate": calculate, "ask_fruit_unit_price": ask_fruit_unit_price}


def query(question, max_turns=5):
    i = 0
    agent = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = agent.invoke(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()  # type: ignore
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f" -- running {action}: {action_input}")
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = f"Observation: {observation}"
        else:
            print(agent.messages)
            return


if __name__ == "__main__":
    question = sys.argv[1:][0]
    query(question)

# [
#     {
#         "role": "system",
#         "content": "You run in a loop of Thought, Action, PAUSE, Observation.\nAt the end of the loop you output an Answer\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the actions available to you - then return PAUSE.\nObservation will be the result of running those actions.\n\nYour available actions are:\n\ncalculate:\ne.g. calculate: 4 * 7 / 3\nRuns a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n\nask_fruit_unit_price:\ne.g. ask_fruit_unit_price: apple\nAsks the user for the price of a fruit\n\nExample session:\n\nQuestion: What is the unit price of apple?\nThought: I need to ask the user for the price of an apple to provide the unit price. \nAction: ask_fruit_unit_price: apple\nPAUSE\n\nYou will be called again with this:\n\nObservation: Apple unit price is 10/kg\n\nYou then output:\n\nAnswer: The unit price of apple is 10 per kg.",
#     },
#     {
#         "role": "user",
#         "content": "What is the total price of 3kg of apple and 2kg of banana?",
#     },
#     {
#         "role": "assistant",
#         "content": "Thought: I need to find the unit prices of apple and banana first, then calculate the total cost for the given quantities.  \nAction: ask_fruit_unit_price: apple  \nPAUSE",
#     },
#     {"role": "user", "content": "Observation: apple   unit price is 20/kg"},
#     {
#         "role": "assistant",
#         "content": "Thought: Now I have the unit price for apple (20/kg). Next, I need the unit price for banana.  \nAction: ask_fruit_unit_price: banana  \nPAUSE",
#     },
#     {"role": "user", "content": "Observation: banana   unit price is 20/kg"},
#     {
#         "role": "assistant",
#         "content": "Thought: Now I have both unit prices: apple is 20/kg and banana is 20/kg. I need to calculate the total price for 3kg of apple and 2kg of banana.  \nAction: calculate: 3 * 20 + 2 * 20  \nPAUSE",
#     },
#     {"role": "user", "content": "Observation: 100"},
#     {
#         "role": "assistant",
#         "content": "Answer: The total price for 3kg of apple and 2kg of banana is 100.",
#     },
# ]
