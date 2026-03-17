import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
from config import QUESTION_PROMPT

class GeminiScorer:
    def __init__(self, api_key: str, base_url: str = ""):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.question = QUESTION_PROMPT
        
    def load_annotations(self, json_path: str) -> Dict[str, str]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = {}
            for item in data:
                video_id = item['video_id']
                sound_type = item['sound_type']
                annotation = item['annotation']
                key = f"{video_id}_{sound_type}"
                annotations[key] = annotation
            
            return annotations
            
        except Exception as e:
            print(e)
            return {}
    
    def extract_video_info(self, filename: str) -> Optional[tuple]:
        match = re.match(r'combined_(\d+)_([^_]+)_combined_result\.json', filename)
        if match:
            video_id = match.group(1)
            sound_type = match.group(2)
            return video_id, sound_type
        return None
    
    def get_gpt4_score(self, question: str, correct_answer: str, predicted_answer: str) -> Optional[Dict]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent chatbot designed for evaluating the audio-visual understanding of generative outputs for audio-visual-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they correctly reflect the event in the audio-visual content."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Please evaluate the following audio-visual-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {correct_answer}\n"
                            f"Predicted Answer: {predicted_answer}\n\n"
                            "Provide evaluations only in the form of audio accuracy and visual accuracy scores, respectively, which are an float value between 0.0 and 5.0, with 5.0 indicating the highest level of consistency between predictions and correct responses. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the temporal accuracy score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'audio score': 2.8, 'visual score': 4.8}."
                        )
                    }
                ],
                temperature=1
            )
            
            response_text = response.choices[0].message.content.strip()            
            try:
                score_dict = eval(response_text)
                if isinstance(score_dict, dict) and 'audio score' in score_dict and 'visual score' in score_dict:
                    return {
                        'audio_score': float(score_dict['audio score']),
                        'visual_score': float(score_dict['visual score']),
                        'raw_response': response_text
                    }
            except:
                pass
            
            audio_match = re.search(r"['\"]audio score['\"]:\s*([0-9.]+)", response_text)
            visual_match = re.search(r"['\"]visual score['\"]:\s*([0-9.]+)", response_text)
            
            if audio_match and visual_match:
                return {
                    'audio_score': float(audio_match.group(1)),
                    'visual_score': float(visual_match.group(1)),
                    'raw_response': response_text
                }
            
            return None
            
        except Exception as e:
            print(e)
            return None
    
    def process_result_file(self, file_path: str, annotations: Dict[str, str]) -> Dict:
        try:
            filename = os.path.basename(file_path)
            
            video_info = self.extract_video_info(filename)
            video_id, sound_type = video_info
            annotation_key = f"{video_id}_{sound_type}"
            correct_answer = annotations[annotation_key]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            result = {
                'filename': filename,
                'video_id': video_id,
                'sound_type': sound_type,
                'correct_answer': correct_answer,
                'flash_result': None,
                'pro_result': None
            }
            
            flash_data = data.get('gemini_flash_result', {})
            if flash_data.get('success', False):
                flash_response = flash_data.get('response', '')
                print(f"flash res: {video_id}_{sound_type}")
                flash_score = self.get_gpt4_score(self.question, correct_answer, flash_response)
                result['flash_result'] = {
                    'response': flash_response,
                    'score': flash_score
                }
                
                time.sleep(1)
            
            pro_data = data.get('gemini_pro_result', {})
            if pro_data.get('success', False):
                pro_response = pro_data.get('response', '')
                print(f"pro res: {video_id}_{sound_type}")
                pro_score = self.get_gpt4_score(self.question, correct_answer, pro_response)
                result['pro_result'] = {
                    'response': pro_response,
                    'score': pro_score
                }
                
                time.sleep(1)
            
            return result
            
        except Exception as e:
            print(e)
            return None
    
    def score_all_results(self, results_folder: str, annotations_file: str, output_file: str = "gemini_scores.json"):
        annotations = self.load_annotations(annotations_file)
        if not annotations:
            return
        
        results_path = Path(results_folder)
        combined_files = list(results_path.glob("*_combined_result.json"))
        
        all_results = []
        flash_scores = {'audio': [], 'visual': []}
        pro_scores = {'audio': [], 'visual': []}
        
        for i, file_path in enumerate(sorted(combined_files), 1):            
            result = self.process_result_file(str(file_path), annotations)
            if result:
                all_results.append(result)
                
                if result['flash_result'] and result['flash_result']['score']:
                    score = result['flash_result']['score']
                    flash_scores['audio'].append(score['audio_score'])
                    flash_scores['visual'].append(score['visual_score'])
                
                if result['pro_result'] and result['pro_result']['score']:
                    score = result['pro_result']['score']
                    pro_scores['audio'].append(score['audio_score'])
                    pro_scores['visual'].append(score['visual_score'])
        
        stats = self.calculate_statistics(flash_scores, pro_scores)
        
        output_data = {
            'metadata': {
                'total_files': len(combined_files),
                'processed_files': len(all_results),
                'question': self.question,
                'annotation_file': annotations_file
            },
            'statistics': stats,
            'detailed_results': all_results
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"saved: {output_file}")
        except Exception as e:
            print(e)
        

        self.print_statistics(stats)
        
        return output_data
    
    def calculate_statistics(self, flash_scores: Dict, pro_scores: Dict) -> Dict:
        def calc_stats(scores):
            if not scores:
                return {
                    'count': 0, 
                    'total_score': 0.0,
                    'max_possible_score': 0.0,
                    'percentage': 0.0,
                    'mean': 0.0, 
                    'min': 0.0, 
                    'max': 0.0
                }
            
            total_score = sum(scores)
            count = len(scores)
            max_possible = 5.0 * count 
            percentage = (total_score / max_possible * 100) if max_possible > 0 else 0
            
            return {
                'count': count,
                'total_score': total_score,
                'max_possible_score': max_possible,
                'percentage': percentage,
                'mean': total_score / count,
                'min': min(scores),
                'max': max(scores)
            }
        
        return {
            'flash': {
                'audio': calc_stats(flash_scores['audio']),
                'visual': calc_stats(flash_scores['visual'])
            },
            'pro': {
                'audio': calc_stats(pro_scores['audio']),
                'visual': calc_stats(pro_scores['visual'])
            }
        }
    
    def print_statistics(self, stats: Dict):

        flash_audio = stats['flash']['audio']
        flash_visual = stats['flash']['visual']
        print(f"\n🔥 Gemini 2.5 Flash:")
        print(f"   Audio Acc.: {flash_audio['total_score']:.1f} / {flash_audio['max_possible_score']:.1f} ({flash_audio['percentage']:.1f}%)")
        print(f"   Visual Acc.: {flash_visual['total_score']:.1f} / {flash_visual['max_possible_score']:.1f} ({flash_visual['percentage']:.1f}%)")
        print(f"   Audio -mean acc: {flash_audio['mean']:.2f} ")
        print(f"   Visual -mean acc: {flash_visual['mean']:.2f} ")
        
        pro_audio = stats['pro']['audio']
        pro_visual = stats['pro']['visual']
        print(f"\n⭐ Gemini 2.5 Pro:")
        print(f"   Audio Acc.: {pro_audio['total_score']:.1f} / {pro_audio['max_possible_score']:.1f} ({pro_audio['percentage']:.1f}%)")
        print(f"   Visual Acc.: {pro_visual['total_score']:.1f} / {pro_visual['max_possible_score']:.1f} ({pro_visual['percentage']:.1f}%)")
        print(f"   Audio -mean acc: {pro_audio['mean']:.2f}")
        print(f"   Visual -mean acc: {pro_visual['mean']:.2f}")
        
        if flash_audio['count'] > 0 and pro_audio['count'] > 0:
            audio_total_diff = pro_audio['total_score'] - flash_audio['total_score']
            visual_total_diff = pro_visual['total_score'] - flash_visual['total_score']
            audio_perc_diff = pro_audio['percentage'] - flash_audio['percentage']
            visual_perc_diff = pro_visual['percentage'] - flash_visual['percentage']
            
            print(f"   Audio Acc.: {audio_total_diff:+.1f} ({audio_perc_diff:+.1f}%)")
            print(f"   Visual Acc.: {visual_total_diff:+.1f} ({visual_perc_diff:+.1f}%)")
            
            audio_mean_diff = pro_audio['mean'] - flash_audio['mean']
            visual_mean_diff = pro_visual['mean'] - flash_visual['mean']
            print(f"   Audio -mean acc: {audio_mean_diff:+.2f} (Pro vs Flash)")
            print(f"   Visual -mean acc: {visual_mean_diff:+.2f} (Pro vs Flash)")
        
        print("="*60)

def main():

    api_key = ""
    results_folder = "video_audio_results"
    annotations_file = "avconfusebench_test_m2.json"
    
    scorer = GeminiScorer(api_key)
    
    result = scorer.score_all_results(results_folder, annotations_file)


if __name__ == "__main__":
    main() 