from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
from datasets import DatasetDict
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from transformers import TrainingArguments
import ast
from trl import GRPOTrainer_stepRR
# from trainer.kd_sft_trainer_test import SFTTrainer
import librosa
from dataclasses import dataclass, field
from typing import Optional
import re
from datetime import datetime
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from math_verify import parse, verify
from transformers import (
    Trainer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen2AudioForConditionalGeneration,
    Qwen2VLProcessor,
    AutoProcessor,
)
import os
import wandb
from trainer.tools import preprocess_func, format_data, format_data_a, format_data_v_nolabel
from trainer.review import compute_reward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["arr_reward", "avc_reward", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )

    dataset_name: Optional[str] = field(
        default='/musicavqa_train_audiovisual_grpo.json',
        metadata={"help": "dataset"},
    )

policy_model_dir = "Qwen/Qwen2.5-Omni-3B/"
ref_model_dir = "Qwen/Qwen2.5-Audio-7B-Instruct/"

processor = AutoProcessor.from_pretrained(policy_model_dir)
processor_ref = AutoProcessor.from_pretrained(ref_model_dir)

def collate_fn(examples):
    texts = processor.apply_chat_template(format_data(examples[0]), tokenize=False) 
    audios_inputs, videos_inputs = torch.load('./Qwen2_Omni_feats/' +examples[0]['video_id'] + '/audio_inputs.pt',weights_only = False), torch.load('./Qwen2_Omni_feats/' +examples[0]['video_id'] + '/video_inputs.pt',weights_only = False)
    USE_AUDIO_IN_VIDEO = True
    batch1 = processor(
        text=texts, audio=audios_inputs, images=None, videos=videos_inputs, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    labels = batch1["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    audio_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.audio_token)]  # Convert image token to ID
    video_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.video_token)]  # Convert image token to ID
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    for audio_token_id in audio_tokens:
        labels[labels == audio_token_id] = -100 
    for video_token_id in video_tokens:
        labels[labels == video_token_id] = -100 
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100 

    batch1["labels"] = labels  # Add labels to the batch
        
    texts_a = [processor_ref.apply_chat_template(format_data_a(example), tokenize=False) for example in examples]
    audio_inputs = [librosa.load(example['file_path2'],sr=processor_ref.feature_extractor.sampling_rate)[0] for example in examples]

    batch2 = processor_ref(
        text=texts_a, audios=audio_inputs, return_tensors="pt", padding=True
    )

    return (batch1, batch2, [format_data_v_nolabel(example) for example in examples]) 


def arr_reward(completions, solution, references, **kwargs):
    contents = [completion for completion in completions]
    audio_thinks = []
    rewards = []
    task = 'Given a reference answer, retrieve semantically similar content'
    for content in contents:
        think_match = re.search(r'<a-think>(.*?)</a-think>', content)
        think_answer = think_match.group(1).strip() if think_match else content.strip()
        audio_thinks.append(think_answer)
    s_arr_group = compute_reward(task,references,audio_thinks)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, s_arr in zip(contents, solution, s_arr_group):
        reward = 0.0
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            ground_truth = ground_truth.replace(' ','').replace('_','').lower()
            student_answer = student_answer.replace(' ','').replace('_','').lower()
            if ground_truth in student_answer or student_answer in ground_truth:
                if s_arr > 0.80:
                    reward = 1.0
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail
        
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} arr_reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
    return rewards

def avc_reward(completions, solution, references, **kwargs):
    contents = [completion for completion in completions]
    rewards = []
    audio_thinks = []
    visual_thinks = []
    task = 'Given a reference answer, retrieve semantically coherent content'
    for content in contents:
        a_think_match = re.search(r'<a-think>(.*?)</a-think>', content)
        a_think_answer = a_think_match.group(1).strip() if a_think_match else content.strip()
        v_think_match = re.search(r'<v-think>(.*?)</v-think>', content)
        v_think_answer = v_think_match.group(1).strip() if v_think_match else content.strip()
        audio_thinks.append(a_think_answer)
        visual_thinks.append(v_think_answer)
    s_avc_group = compute_reward(task,visual_thinks,audio_thinks)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol, s_avc in zip(contents, solution, s_avc_group):
        reward = 0.0
        try:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            ground_truth = ground_truth.replace(' ','').replace('_','').lower()
            student_answer = student_answer.replace(' ','').replace('_','').lower()
            if student_answer is not None:
                if ground_truth in student_answer or student_answer in ground_truth:
                    reward = 1.0 + round(s_avc, 2)
                else:
                    reward = round(s_avc, 2)
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} avc_reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern1, pattern2, pattern3, pattern4, pattern5, pattern6 = "<a-think>","</a-think>","<v-think>","</v-think>","<answer>","</answer>"
    matches = [ ]
    for completion in completions:
        if pattern1 in completion and pattern2 in completion and pattern3 in completion and pattern4 in completion and pattern5 in completion and pattern6 in completion:
            matches.append(True)
        else:
            matches.append(None)
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "arr_reward": arr_reward,
    "avc_reward": avc_reward,
    "format": format_reward,
}


# Configure training arguments
training_args = GRPOConfig(
    output_dir="./output/trainer_output_visual_RL_CoMM",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=32,  # Batch size for training
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=1e-6,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    logging_steps=10,  # Steps interval for logging
    save_steps=20,  # Steps interval for saving
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    report_to="wandb",  # Reporting tool for tracking metrics
    gradient_checkpointing_kwargs={"use_reentrant": True},  # Options for gradient checkpointing
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset


def main(script_args, training_args, model_args):
    training_args.policy_model_dir = policy_model_dir
    training_args.ref_model_dir = ref_model_dir

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        training_args.policy_model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
    _apply_liger_kernel_to_instance(model=model)

    model_audio = Qwen2AudioForConditionalGeneration.from_pretrained(
        training_args.ref_model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor_ref = AutoProcessor.from_pretrained(training_args.ref_model_dir)

    model.enable_input_require_grads()

    dataset = Dataset.from_json("./musicavqa_train.json")
    dataset = dataset.select(range(1000))
    train_dataset = dataset.map(preprocess_func)
    reward_funcs_names = ["arr_reward", "avc_reward", "format"]
    reward_funcs = [reward_funcs_registry[func] for func in reward_funcs_names]

    trainer = GRPOTrainer_stepRR(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        ref_model=model_audio,
        audio_processing_class=processor_ref,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

def get_args():
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    return parser.parse_args_and_config()  

def train():
    script_args, training_args, model_args = get_args()
    main(script_args, training_args, model_args)