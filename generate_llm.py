import os
import sys
from fire import Fire
import gradio as gr
import torch
import emoji
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():  device = "mps"
except:
    pass  # noqa: E722


# server_name: str = "0.0.0.0", Allows to listen on all interfaces by providing '0.
def main(load_8bit: bool = True, base_model: str = "../../v2/7B",
         lora_weights: str = "./output/steganography_finetune/7B-styleme-8-qv",
         prompt_template: str = "", share_gradio: bool = True):

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    if device == "cuda:0":
        model = LlamaForCausalLM.from_pretrained(base_model, load_in_8bit=load_8bit, torch_dtype=torch.float16, device_map="auto")
        # model = PeftModel.from_pretrained(model, lora_weights, torch_dtype=torch.float16)
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(base_model, device_map={"": device}, torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, lora_weights, device_map={"": device}, torch_dtype=torch.float16)
    else:
        model = LlamaForCausalLM.from_pretrained(base_model, device_map={"": device}, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(model, lora_weights, device_map={"": device})

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:  model.half()  # seems to fix bugs for some users.
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":  model = torch.compile(model)

    def evaluate(
            instruction,
            input=None,
            temperature=0.7, 
            top_p=0.75, 
            top_k=40, 
            num_beams=3, 
            max_new_tokens=256, 
            stream_output=False,
            do_sample=False,
            early_stopping=True,
            **kwargs
    ):

        prompt = prompter.generate_prompt(instruction, input) 
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams,
                                             early_stopping=early_stopping, do_sample=do_sample, **kwargs)
        # GenerationConfig
        generate_params = {"input_ids": input_ids,
                           "generation_config": generation_config,
                           "return_dict_in_generate": True,
                           "output_scores": True,
                           "max_new_tokens": max_new_tokens}

        if stream_output: 
            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault("stopping_criteria", transformers.StoppingCriteriaList())
                kwargs["stopping_criteria"].append(Stream(callback_func=callback))
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
                    if output[-1] in [tokenizer.eos_token_id]: break
                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # -------------- Without streaming --------------
        with torch.no_grad():
            
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,  # temperature, top_k
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
                output_hidden_states=True,
                max_new_tokens=max_new_tokens,
            )
            input_ids = generation_output.sequences

        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        yield prompter.get_response(output)  
        print(output)

    # gradio_interface
    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Instruction", placeholder="Tell me about alpacas."),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=0.7, label="Temperature (Larger values increase the diversity of generation.)"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p (Larger values produce better results, but also increase the computational burden.)"),
            gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k (Specify the top_k words with the highest probability.)"),
            gr.components.Slider(minimum=1, maximum=4, step=1, value=1, label="Beams (Select num_beams from the top_k words as candidate sequences. Larger values produce better results.)"),
            gr.components.Slider(minimum=1, maximum=2000, step=1, value=512, label="Max tokens (Longest sequence.)"),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[gr.inputs.Textbox(lines=5, label="Output")],
        title="STEGPT", description="STEGPT is a model that embeds secret information.",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)


if __name__ == "__main__":
    Fire(main)
