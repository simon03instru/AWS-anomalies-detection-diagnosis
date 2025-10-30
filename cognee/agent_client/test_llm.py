from ollama import Client

model = "gpt-oss:20b"
host = "http://10.33.205.34:11112"

messages = [
    {"role": "user", "content": """Solve the math problem step by step and provide the final answer.
What is 35 + 27 * 8?"""},
]

client = Client(host=host)

options =  {
    "temperature": 0,
    "num_ctx": 8192
}
stream = client.chat(model=model, messages=messages, stream=True, options=options)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)