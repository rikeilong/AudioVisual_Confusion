import ast

def preprocess_func(sample):
    question = sample['question_content'].rstrip().split(' ')
    question[-1] = question[-1][:-1]
    p = 0
    for pos in range(len(question)):
        if '<' in question[pos]:
            question[pos] = ast.literal_eval(sample['templ_values'])[p]
            p += 1
    
    input_content = ' '.join(question)
    output_content = sample["anser"]
    file_path1 = './Qwen2_Omni_feats/' + sample['video_id']
    file_path2 = '/MusicAVQA/video/' + sample['video_id'] + '.mp4'
    return {
        "text": input_content,
        "file_path1": file_path1,
        "file_path2": file_path2,
    }

def format_data(sample):
    return [        
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["file_path1"] ,
                },
                {
                    "type": "text",
                    "text": "This is a video question. Please answer the question based on the video.\n {}? \nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>species name</answer>\nPlease strictly follow the format.".format(sample['text'])
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["anser"]}],
        },
    ]

def format_data_v_question(sample):
    return [        
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["file_path1"] ,
                },
                {
                    "type": "text",
                    "text": sample['text']
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["anser"]}],
        },

    ]

def format_data_v_nolabel(sample):
    return [        
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": sample["file_path1"] ,
                },
                {
                    "type": "text",
                    "text": "This is a video question. Please answer the question based on the video.\n {}? \nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output answer format should be as follows:\n<think> ... </think> <answer>species name</answer>\nPlease strictly follow the format.".format(sample['text'])
                },
            ],
        },
    ]

def format_data_a(sample):
    return [        
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": sample["file_path2"] ,
                },
                {
                    "type": "text",
                    "text": "You are an assistant to help with hearing. Here is the answer '{answer}' to the question '{question}'. And your task is: Reasoning what you hear that is useful for answering the question in <a-think> </a-think>. The output format should be as follows: <a-think>...</a-think>".format(answer=sample['anser'],question=sample['text']),
                },
            ],
        }

    ]
