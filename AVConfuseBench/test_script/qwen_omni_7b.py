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
    eval_dataset = json.load(open("./datasets/AVConfuseBench/avconfusebench_test_m1.json"))

    total = len(eval_dataset)
    corrects = []

    generated_answers = []
    for data in tqdm(eval_dataset):
        
        file_path = data['audio']
        answer = data['answer']
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{file_path}",
                    },
                    {"type": "text", "text": data['question']},
                ],
            }
        ]
        USE_AUDIO_IN_VIDEO = True
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio
        try:
            text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, text_ids)]
            output_text = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            pred = output_text[0].lower()
            if pred in answer or answer in pred:
                corrects.append(1)
                yes_no = 1
            else:
                corrects.append(0)
                yes_no = 0
                
        except Exception as e:
            pred = ''
            continue

        data['output'] = output_text[0].lower()
        data['yes/no'] = yes_no
        generated_answers.append(data)
        
        with open('./datasets/AVConfuseBench/output/omni_7B.json', 'w') as json_file:
            json.dump(generated_answers, json_file, indent=4, ensure_ascii=False)

    print('Total Accuracy: %.2f %%' % (
            100 * sum(corrects) / len(eval_dataset)))
    
    print('Yes(%): %.2f %%' % (
            100 - (100 * sum(corrects) / len(eval_dataset))))

if __name__ == "__main__":
    inference()