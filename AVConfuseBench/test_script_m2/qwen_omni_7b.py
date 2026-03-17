import soundfile as sf
import json
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import ast
import torch

model_dir = '/Qwen/Qwen2.5-Omni-7B/'

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype="bfloat16",
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)

def inference():
    eval_dataset = json.load(open("./datasets/AVConfuseBench/avconfusebench_test_m2.json"))

    generated_answers = []
    for data in tqdm(eval_dataset):
        
        file_path = data['filename']
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{file_path}",
                    },
                    {"type": "text", "text": "Describe what you see and what you hear. The output format should be as follows: Based on the video...\nBased on the audio..."},
                ],
            }
        ]
        USE_AUDIO_IN_VIDEO = True
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        try:
            text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,max_tokens=256)
            trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, text_ids)]
            output_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                
        except Exception as e:
            output_text = ''
            continue

        data['output'] = output_text
        generated_answers.append(data)
        
        with open('./datasets/AVConfuseBench/output/omni_7B_m2.json', 'w') as json_file:
            json.dump(generated_answers, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    inference()