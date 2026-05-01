"""
CISC 886 — Cloud Computing — Group 24
Smart Banking AI — Gradio web interface

Loads the merged F16 GGUF model (qwen2.5-1.5b-instruct.F16.gguf)
via llama-cpp-python and exposes a Gradio ChatInterface on
0.0.0.0:7860 with a *.gradio.live share tunnel.

Auto-started at boot by /etc/systemd/system/banking-ui.service.
"""

from llama_cpp import Llama
import gradio as gr

print("Loading the GGUF model... Please wait.")

llm = Llama(
    model_path="qwen2.5-1.5b-instruct.F16.gguf",
    n_ctx=2048,
)


def chat_with_bot(message, history):
    formatted_prompt = ""
    for human, assistant in history:
        formatted_prompt += (
            f"<|im_start|>user\n{human}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        )
    formatted_prompt += (
        f"<|im_start|>user\n{message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    response = llm(
        formatted_prompt,
        max_tokens=256,
        stop=["<|im_end|>"],
        echo=False,
    )
    return response["choices"][0]["text"]


demo = gr.ChatInterface(
    chat_with_bot,
    title="🏦 Smart Banking AI - Group 24",
    description="Welcome! I am the bank's smart assistant. How can I help you today?",
)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
