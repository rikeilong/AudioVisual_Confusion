import torch
import vllm
from vllm import LLM

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def compute_reward(task, references, queries, **kwargs):
    references = [get_detailed_instruct(task, reference) for reference in references]
    queries = [query for query in queries]

    input_texts = references + queries
    model = LLM(model="./Qwen3-Embedding-0.6B/", task="embed")
    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T)
    return scores.tolist()[0]