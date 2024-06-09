import os
import sys
from fire import Fire
import gradio as gr
import torch
import emoji
import transformers
from peft import PeftModel
import torch.nn.functional as F
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter


os.environ['NUMEXPR_MAX_THREADS'] = '32'
import time

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():  device = "mps"
except:
    pass


def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i


def near(alist, anum):
    up = len(alist) - 1
    if up == 0:
        return 0
    bottom = 0

    while up - bottom > 1:
        index = int((up + bottom) / 2)

        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index

    if up - bottom == 1:

        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up

    return index


def main(load_8bit: bool = True, base_model: str = "./LLMs/LLaMA2-7B",
         name: str = "./7B-styleme-8-qv",
         lora_weights: str = "./ft-model/7B-8qv",
         prompt_template: str = "",
         share_gradio: bool = True
         ):

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    if device == "cuda:0":
        model = LlamaForCausalLM.from_pretrained(base_model,
                                                 load_in_8bit=load_8bit,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")

        model = PeftModel.from_pretrained(model,
                                          lora_weights,
                                          torch_dtype=torch.float16)

    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(base_model,
                                                 device_map={"": device},
                                                 torch_dtype=torch.float16)

        model = PeftModel.from_pretrained(model,
                                          lora_weights,
                                          device_map={"": device},
                                          torch_dtype=torch.float16)

    else:
        model = LlamaForCausalLM.from_pretrained(base_model,
                                                 device_map={"": device},
                                                 low_cpu_mem_usage=True)

        model = PeftModel.from_pretrained(model,
                                          lora_weights,
                                          device_map={"": device})

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":  model = torch.compile(model)


    def evaluate(
            instruction,
            input=None,
            temperature=0.7,
            top_p=0.75,
            top_k=100,
            num_beams=1,
            max_new_tokens=512,
            stream_output=False,
            count=8,
            PRECISION=48,
            map="linear",  # sqrt / sqrt3 / linear
            **kwargs
    ):

        prompt = prompter.generate_prompt(instruction, input)  # prompt包含一个instruction一个input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(temperature=temperature,
                                             top_p=top_p,
                                             top_k=top_k,
                                             num_beams=num_beams,
                                             early_stopping=True,
                                             do_sample=False,
                                             **kwargs
                                             )

        # GenerationConfig
        generate_params = {"input_ids": input_ids,
                           "generation_config": generation_config,
                           "return_dict_in_generate": True,
                           "output_scores": True,
                           "max_new_tokens": max_new_tokens}

        # -------------- Without streaming --------------
        with open('pngSteg/bit_stream.txt', 'r', encoding='utf8') as f:
            bit_stream = f.read().strip()
            bit_stream += bit_stream
        # bit_stream = '0001010000100100000101010001010010010'
        bit_index = int(torch.randint(0, high=1000, size=(1,)))  # or 0

        with torch.no_grad():
            start = time.time()
            stega_text, stega_bits = [], ''

            # PRECISION = 16
            # max_val = 2 ** PRECISION  # num of intervals
            # cur_interval = [0, max_val]  # bottom inclusive, top exclusive
            for i in range(max_new_tokens - 1):
                if '</s>' in stega_text:
                    break

                generation_output1 = model(input_ids)
                log_prob = generation_output1.logits
                prob = torch.softmax(log_prob, dim=-1)[:, -1, :].reshape(-1)
                # prob[1] = 0  # set unk to zero
                prob = prob / prob.sum()
                prob, indices = prob.sort(descending=True)
                # start recursion
                bit_tmp = 0
                PRECISION = PRECISION
                max_val = 2 ** PRECISION  # num of intervals
                cur_interval = [0, max_val]  # bottom inclusive, top exclusive
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range

                if prob[-1] < cur_threshold:
                    k = max(2, (prob < cur_threshold).nonzero()[0].item())
                    prob = prob[:k]
                    indices = indices[:k]

                prob = prob[:top_k]
                indices = indices[:top_k]

                if map == "sqrt":
                    prob = torch.round(torch.sqrt(prob), decimals=4)
                elif map == "sqrt3":
                    prob = torch.pow(prob, 1 / 3)
                elif map == "sqrt4":
                    prob = torch.pow(prob, 1 / 4)
                prob = prob / prob.sum()
                prob = prob.double()
                prob *= cur_int_range
                prob = prob.round().long()

                cum_probs = prob.cumsum(0)
                overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])

                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]
                cum_probs += cur_int_range - cum_probs[-1]

                cum_probs += cur_interval[0]

                message_bits = bit_stream[bit_index: bit_index + PRECISION]
                message_bits = [int(_) for _ in message_bits]
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]
                #
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, PRECISION)))
                new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, PRECISION)))
                #
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)

                prev = indices[selection].view(1, 1)

                gen = int(prev)
                input_ids = torch.cat([input_ids, torch.LongTensor([[gen]]).to(device)], dim=1).to(device)
                stega_bits += bit_stream[bit_index:bit_index + num_bits_encoded]
                bit_index += num_bits_encoded

                # if gen == 29889:
                #     break
                if gen == 29889:
                    # print(f"{gen},{tokenizer.decode(gen)}")
                    count -= 1
                    if 0 == count:
                        break

        # scores = generation_output.scores
        # s = generation_output.sequences[0]
        end = time.time()
        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # yield prompter.get_response(output)
        costTime = end-start
        print(stega_bits)
        return output, stega_bits, costTime

    import string
    import csv

    def count_words(s):
        for char in string.punctuation:
            s = s.replace(char, '')
        words = s.split()
        return len(words)

    Instruction = []
    iter = int(500)
    last_part = os.path.basename(name)
    if last_part == "P_Shakespeare":
        Instruction = ["Introduction to Shakespeare's writing style.",
                       # "Introduction to Dickens's early life."
                       # "Generate a text in the style of Shakespeare.",
                       # "You are a writer who knows Shakespeare very well. Generate a text in the style of Shakespeare.",
                       ]

    elif last_part == "P_Andersen":
        Instruction = ["Generate a text in the style of Andersen.",
                       "You are a writer who knows Andersen very well. Generate a text in the style of Andersen.",
                       ]

    elif last_part == "P_Conan":
        Instruction = ["Generate a text in the style of Conan.",
                       "You are a writer who knows Conan very well. Generate a text in the style of Conan.",
                       ]

    elif last_part == "P_Dickens":
        Instruction = ["Generate a text in the style of Dickens.",
                       "You are a writer who knows Dickens very well. Generate a text in the style of Dickens.",
                       ]

    elif last_part == "P_Tolstoy":
        Instruction = ["Generate a text in the style of Tolstoy.",
                       "You are a writer who knows Tolstoy very well. Generate a text in the style of Tolstoy.",
                       ]

    elif last_part == "G_movie":
        Instruction = ["Write a general overview of quantum computing"]

    elif last_part == "G_news":
        Instruction = ["Generate the contents in the style of news."]

    elif last_part == "G_twitter":
        Instruction = ["Generate the contents in the style of tweets."]

    elif last_part == "Z_Katyperry":
        Instruction = ["Generate a text in the style of Katyperry."]

    else:
        last_p = last_part.split("T_")[1]
        Instruction = ["Generate the content about the " + last_p + " theme."]

    csv_file = name+".csv"
    row = ["Instruction", "Response", "stega_bits", "costTime(s)", "words", "bits", "Embedding rate"]

    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    for instr in Instruction:

        for j in range(iter):
            print(last_part, ":", j, "/", iter)
            output, stega_bits, costTime = evaluate(instruction=instr)

            output = output.split("Response:\n", 1)[1]
            words = count_words(output)
            chars = len(stega_bits)
            row = [instr,
                   output,
                   stega_bits,
                   costTime,
                   words,
                   chars,
                   chars/words
                   ]

            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)


if __name__ == "__main__":
    names = [
             "P_Shakespeare", "P_Andersen", "P_Conan",
             "P_Dickens", "P_Tolstoy", "G_movie", "G_news", "G_twitter",
             "T_administration", "T_animals", "T_anime",
             "T_architecture", "T_art", "T_astronomy",
             "T_automotive", "T_beauty", "T_biology", "T_chemistry",
             "T_civil", "T_computer_science", "T_consumption", "T_culture", "T_current_affairs",
             "T_dance", "T_design", "T_drinks", "T_earth_sciences", "T_economy", "T_education", "T_emotion",
             "T_engineering", "T_entertainment", "T_family", "T_finance", "T_food", "T_game", "T_geography", "T_gourmet", "T_health", "T_history", "T_humanities", "T_interpersonal",
             "T_judicial", "T_language", "T_law", "T_life", "T_literature", "T_love", "T_mathematics", "T_medical", "T_military", "T_movies", "T_music", "T_nature", "T_pets", "T_philosophy",
             "T_physics", "T_plants", "T_politics", "T_psychology", "T_real_estate", "T_regulations", "T_religion", "T_shopping", "T_society", "T_sport", "T_technology", "T_TV", "T_war", "T_work"
             ]
    for weight in names:
        main(name="./data_stego/a48-sqrt4-100/" + weight)
