import os
import json
import torch
import argparse
from datasets import Dataset, DatasetDict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from tqdm import tqdm
import ast
import multiprocessing as mp

MODEL_NAME = ''

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess video dataset for Qwen-VL model")
    parser.add_argument("--model_name", type=str, default="/Qwen2-VL-7B-Instruct/",
                        help="Path to the pretrained model")
    parser.add_argument("--dataset", type=str, default="charades",
                        help="Dataset name to be preprocessed")
    parser.add_argument("--train_data_path", type=str, default="/musicavqa_train.json",
                        help="Path to the training data JSON file")
    parser.add_argument("--eval_data_path", type=str, default="./Charades/charades_annotation/val.json",
                        help="Path to the evaluation data JSON file")
    parser.add_argument("--video_folder", type=str, default="/MusicAVQA/video/",
                        help="Path to the folder containing video files")
    parser.add_argument("--output_dir", type=str, default='/Qwen2VL_feats/',
                        help="Output directory path. If None, it will be created based on dataset and max_pix values")
    parser.add_argument("--max_pix_size", type=int, default=3584,
                        help="Maximum pixel size")
    parser.add_argument("--min_pix_size", type=int, default=16,
                        help="Minimum pixel size")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of worker processes for multiprocessing")
    
    return parser.parse_args()

def preprocess_single_video(task_args): # Accept task arguments as a single tuple/list
    video_path, processor, max_pixels, min_pixels, example_output_dir, sentence, solution = task_args # Unpack task args
    try:
        if os.path.exists(example_output_dir) == False:
            image_inputs, video_inputs, video_kwargs, fps_inputs = preprocess_video_inner(
                video_path, processor, max_pixels, min_pixels
            )    
            os.makedirs(example_output_dir, exist_ok=True)
            # torch.save(image_inputs, os.path.join(example_output_dir, "image_inputs.pt"))
            torch.save(video_inputs, os.path.join(example_output_dir, "video_inputs.pt"))
            with open(os.path.join(example_output_dir, "video_kwargs.json"), 'w') as f:
                json.dump(video_kwargs, f)

            return {
                "problem": sentence,
                "solution": solution,
                "preprocessed_path": example_output_dir,
                "status": "success"
            }
        else:
            return {
                "problem": sentence,
                "solution": solution,
                "preprocessed_path": example_output_dir,
                "status": "success"
            }
    except Exception as e:
        print(f"Warning: Preprocessing failed for video {video_path}, skipping. Error: {e}")
        return {
            "video_path": video_path,
            "status": "failed",
            "error": str(e)
        }


def process_feats(path):
    failed_video = []
    output_split_dir = '/data/Qwen2VL_feats/'
    files = os.listdir(path)
    for video_file in tqdm(files):
        #video
        video_id = video_file[:-4]
        video_path = os.path.join(path + video_file)
        example_output_dir = os.path.join(output_split_dir, f"{video_id}")
        max_pixels = 3584 * 28 * 28
        min_pixels = 16 * 28 * 28
        messages = [
            {"role": "user", "content": [
                    {"type": "video", 
                    "video": video_path, 
                    "total_pixels": max_pixels, 
                    "min_pixels": min_pixels,
                    },
                ]
            },
        ]
        if os.path.exists(example_output_dir) == False:    
            try:
                image_inputs, video_inputs = process_vision_info(messages)   
                os.makedirs(example_output_dir, exist_ok=True)
                # torch.save(image_inputs, os.path.join(example_output_dir, "image_inputs.pt"))
                torch.save(video_inputs, os.path.join(example_output_dir, "video_inputs.pt"))
            except Exception as e:
                print(f"Warning: Preprocessing failed for video {video_path}, skipping. Error: {e}")
                failed_video.append(video_path)
        else:
            print('is exist')
    print(failed_video)

def process_feats_omni(path):
    from qwen_omni_utils import process_mm_info
    failed_video = []
    output_split_dir = '/data/Qwen2_Omni_feats/'
    files = os.listdir(path)
    for video_file in tqdm(files):
        #video
        video_id = video_file[:-4]
        video_path = os.path.join(path + video_file)
        example_output_dir = os.path.join(output_split_dir, f"{video_id}")
        messages = [
            {"role": "user", "content": [
                    {"type": "video", 
                    "video": video_path, 
                    },
                ]
            },
        ]
        if os.path.exists(example_output_dir) == False:    
            try:
                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
                os.makedirs(example_output_dir, exist_ok=True)
                # torch.save(image_inputs, os.path.join(example_output_dir, "image_inputs.pt"))
                torch.save(audios, os.path.join(example_output_dir, "audio_inputs.pt"))
                torch.save(videos, os.path.join(example_output_dir, "video_inputs.pt"))
            except Exception as e:
                print(f"Warning: Preprocessing failed for video {video_path}, skipping. Error: {e}")
                failed_video.append(video_path)
        else:
            print('is exist')
    print(failed_video)

def preprocess_input_qwen2(path):

    model_dir = '/Qwen2-VL-7B-Instruct/'

    processor = AutoProcessor.from_pretrained(model_dir)
    files = os.listdir(path)
    input_content = 'ssss'
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_content + '?'},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    for video_file in tqdm(files):
        video_path = os.path.join(path + video_file)
        # size = os.path.getsize(os.path.join(video_path,'video_inputs.pt'))
        # if size < 500000:
        #     failed_name.append(video_path)
        # print(failed_name)
        failed_video = []
        if os.path.exists(os.path.join(video_path, "pixel_values_videos.pt")) == False or os.path.exists(os.path.join(video_path, "video_grid_thw.pt")) == False:
            try:
                video_inputs = torch.load(os.path.join(video_path,'video_inputs.pt'))   
                inputs = processor(
                    text=[text],
                    # images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.tolist() for key, value in inputs.items()}
                torch.save(torch.tensor(inputs['pixel_values_videos']), os.path.join(video_path, "pixel_values_videos.pt"))
                torch.save(torch.tensor(inputs['video_grid_thw']).squeeze(0), os.path.join(video_path, "video_grid_thw.pt"))
            except Exception as e:
                print(f"Warning: Preprocessing failed for video {video_path}, skipping. Error: {e}")
                failed_video.append(video_path)
                print(failed_video)
  
def preprocess_video_inner(video_path, processor, max_pixels, min_pixels):
    messages = [
        {"role": "user", "content": [
                {"type": "video", 
                "video": video_path, 
                "total_pixels": max_pixels, 
                "min_pixels": min_pixels,
                },
            ]
        },
    ]
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    
    return image_inputs, video_inputs, video_kwargs, fps_inputs

def process_split(file_path, split_name, video_folder, output_dir, max_pixels, min_pixels, processor, num_workers=8):
    output_split_dir = os.path.join(output_dir, split_name)
    os.makedirs(output_split_dir, exist_ok=True)

    with open(file_path, 'r') as f:
        data = json.load(f)  

    examples = []
    tasks = []
 
    for d in data:
        #question
        question_id = d['question_id']
        question = d['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]
        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(d['templ_values'])[p]
                p += 1
                
        sentence = ' '.join(question)

        #answer
        solution = d['anser']

        #video
        video_id = d['video_id']
        video_path = video_folder + d['video_id'] + '.mp4'
        example_output_dir = os.path.join(output_split_dir, f"{video_id}")

        tasks.append((video_path, processor, max_pixels, min_pixels, example_output_dir, sentence + '?', solution)) # Prepare task arguments as tuples
    
    pbar = tqdm(total=len(tasks), desc=f"Preprocessing {split_name} split") # Initialize progress bar in main process

    with mp.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(preprocess_single_video, tasks) # Use imap_unordered for unordered results, potentially faster

        successful_examples = []
        failed_count = 0
        for result in results: # Iterate through results to update progress bar
            pbar.update(1)
            if result['status'] == 'success':
                successful_examples.append(result)
            else:
                failed_count += 1
                # Optionally log failed videos and errors

    pbar.close() # Close progress bar after processing

    print(f"Preprocessing for split '{split_name}' finished. Failed videos: {failed_count}, Successful videos: {len(successful_examples)}")

    return Dataset.from_list(successful_examples)


def preprocess_dataset_and_save(train_data_path, eval_data_path, video_folder, output_dir, max_pixels, min_pixels, num_workers=8):

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = process_split(train_data_path, "train", video_folder, output_dir, max_pixels, min_pixels, processor, num_workers)
    # eval_dataset = process_split(eval_data_path, "eval", video_folder, output_dir, max_pixels, min_pixels, processor, num_workers)
    # return DatasetDict({"train": train_dataset, "eval": eval_dataset})
    return DatasetDict({"train": train_dataset})


if __name__ == "__main__":
    # args = parse_args()
    # MODEL_NAME = args.model_name
    
    # # Calculate pixel values
    # max_pixels = args.max_pix_size * 28 * 28
    # min_pixels = args.min_pix_size * 28 * 28
    
    # # Setup output directory
    # if args.output_dir is None:
    #     output_dir = f"./{args.dataset}_preprocessed_data_maxpix_{args.max_pix_size}"
    # else:
    #     output_dir = args.output_dir
        
    # print('output_dir', output_dir)

    # dataset_dict = preprocess_dataset_and_save(
    #     args.train_data_path, args.eval_data_path, args.video_folder, 
    #     output_dir, max_pixels, min_pixels, num_workers=args.num_workers
    # )
    
    # print("Preprocessing complete. Datasets saved to:", output_dir)
    # print(dataset_dict)

    # process_feats('/AVHBench/videos/')
    # preprocess_input_qwen2('/MusicAVQA/Qwen2VL_feats/')

    process_feats_omni('...')
    # preprocess_input_qwen2('/AVHBench/Qwen2VL_feats/')