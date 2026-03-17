import os
import json
import time
import base64
import cv2
from pathlib import Path
import requests
from typing import Dict, Any, List, Optional
from moviepy.editor import VideoFileClip

class VideoProcessor:
    """
    Video processor for confused video benchmark evaluation using Gemini models
    Handles two tasks:
    1. confused_video: specific questions from avconfusebench_test.json
    2. confused_video_m2: fixed descriptive question
    """
    
    def __init__(self, api_key: str, base_url: str ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.temp_audio_dir = "temp_audio"
        os.makedirs(self.temp_audio_dir, exist_ok=True)
        
        # Fixed question for confused_video_m2
        self.fixed_question = ("Describe what you see and what you hear. "
                              "The output format should be as follows: "
                              "Based on the video, ...\nBased on the audio, ...")
    
    def load_questions(self, json_path: str) -> Dict[str, List[Dict]]:
        """Load questions from avconfusebench_test.json grouped by video_id"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        grouped_data = {}
        for item in data:
            video_id = item['video_id']
            if video_id not in grouped_data:
                grouped_data[video_id] = []
            grouped_data[video_id].append(item)
        
        return grouped_data
    
    def find_video_file(self, video_folder: str, audio_path: str) -> Optional[str]:
        """Find corresponding video file based on audio path from JSON"""
        audio_filename = os.path.basename(audio_path)
        video_path = os.path.join(video_folder, audio_filename)
        return video_path if os.path.exists(video_path) else None
    
    def extract_video_frames(self, video_path: str, seconds_per_frame: int = 3) -> List[str]:
        """Extract video frames and convert to base64"""
        base64_frames = []
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps * seconds_per_frame)
        curr_frame = 0

        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success:
                break
            
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                frame = cv2.resize(frame, (800, int(height * scale)))
            
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip

        video.release()
        return base64_frames
    
    def extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video"""
        base_name = Path(video_path).stem
        audio_path = os.path.join(self.temp_audio_dir, f"{base_name}.mp3")
        
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            return None
        
        video.audio.write_audiofile(
            audio_path,
            bitrate="32k",
            verbose=False,
            logger=None
        )
        video.close()
        return audio_path
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper API"""
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': audio_file,
                'model': (None, 'whisper-1'),
                'response_format': (None, 'json')
            }
            
            headers_transcribe = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers_transcribe,
                files=files,
                timeout=120
            )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('text', '')
        return "Audio transcription failed"
    
    def analyze_with_gemini(self, model_name: str, base64_frames: List[str], 
                           audio_transcription: str, question: str, audio_path: str = None) -> Dict[str, Any]:
        """Analyze video with Gemini model"""
        content = [{"type": "text", "text": question}]

        # Add video frames
        for frame in base64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })

        # Add audio directly if available, otherwise use transcription
        if audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, 'rb') as audio_file:
                    audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
                    content.append({
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/mp3;base64,{audio_data}"}
                    })
            except Exception:
                content.append({
                    "type": "text",
                    "text": f"Audio transcription: {audio_transcription}"
                })
        else:
            content.append({
                "type": "text",
                "text": f"Audio transcription: {audio_transcription}"
            })

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "response": result["choices"][0]["message"]["content"],
                "model": model_name
            }
        else:
            return {
                "success": False,
                "error": f"API error {response.status_code}: {response.text}",
                "model": model_name
            }
    
    def process_question(self, video_path: str, question_data: Dict) -> Dict[str, Any]:
        """Process a single question for Task 1 (confused_video)"""
        base64_frames = self.extract_video_frames(video_path)
        
        # Extract and transcribe audio
        audio_path = self.extract_audio(video_path)
        if audio_path:
            raw_transcription = self.transcribe_audio(audio_path)
            
            # Filter meaningless transcriptions
            cleaned_text = ''.join(c for c in raw_transcription if c.isalnum() or c.isspace())
            is_meaningless = (
                len(cleaned_text.strip()) < 10 or
                len(set(raw_transcription.split())) < 3
            )
            
            audio_transcription = ("The audio consists mainly of music or background sounds "
                                 "with no clear spoken language content" if is_meaningless 
                                 else raw_transcription)
        else:
            audio_transcription = "This video has no audio track"
        
        # Analyze with both Gemini models
        flash_result = self.analyze_with_gemini(
            "gemini-2.5-flash", base64_frames, audio_transcription, 
            question_data['question'], audio_path
        )
        
        pro_result = self.analyze_with_gemini(
            "gemini-2.5-pro", base64_frames, audio_transcription, 
            question_data['question'], audio_path
        )
        
        # Clean up temporary audio file
        if audio_path:
            try:
                os.remove(audio_path)
            except:
                pass
        
        return {
            "video_path": video_path,
            "video_id": question_data['video_id'],
            "question": question_data['question'],
            "expected_answer": question_data['answer'],
            "instruments": question_data['instruments'],
            "frames_extracted": len(base64_frames),
            "audio_description": audio_transcription,
            "gemini_flash_result": flash_result,
            "gemini_pro_result": pro_result
        }
    
    def process_video_description(self, video_path: str) -> Dict[str, Any]:
        """Process a single video for Task 2 (confused_video_m2)"""
        base64_frames = self.extract_video_frames(video_path)
        
        # Extract and transcribe audio
        audio_path = self.extract_audio(video_path)
        if audio_path:
            raw_transcription = self.transcribe_audio(audio_path)
            
            # Filter meaningless transcriptions
            cleaned_text = ''.join(c for c in raw_transcription if c.isalnum() or c.isspace())
            is_meaningless = (
                len(cleaned_text.strip()) < 10 or
                len(set(raw_transcription.split())) < 3
            )
            
            audio_transcription = ("The audio consists mainly of music or background sounds "
                                 "with no clear spoken language content" if is_meaningless 
                                 else raw_transcription)
        else:
            audio_transcription = "This video has no audio track"
        
        # Analyze with both Gemini models using fixed question
        flash_result = self.analyze_with_gemini(
            "gemini-2.5-flash", base64_frames, audio_transcription, 
            self.fixed_question, audio_path
        )
        
        pro_result = self.analyze_with_gemini(
            "gemini-2.5-pro", base64_frames, audio_transcription, 
            self.fixed_question, audio_path
        )
        
        # Clean up temporary audio file
        if audio_path:
            try:
                os.remove(audio_path)
            except:
                pass
        
        return {
            "video_path": video_path,
            "question": self.fixed_question,
            "frames_extracted": len(base64_frames),
            "audio_description": audio_transcription,
            "gemini_flash_result": flash_result,
            "gemini_pro_result": pro_result
        }
    
    def process_task1_videos(self, video_folder: str, json_path: str, output_folder: str = "task1_results"):
        """Process Task 1: confused_video with specific questions"""
        os.makedirs(output_folder, exist_ok=True)
        
        questions_data = self.load_questions(json_path)
        total_questions = sum(len(questions) for questions in questions_data.values())
        processed_count = 0
        
        for video_id, questions in questions_data.items():
            for i, question_data in enumerate(questions, 1):
                video_file = self.find_video_file(video_folder, question_data['audio'])
                
                if not video_file:
                    continue
                
                # Generate output filename
                audio_filename = os.path.basename(question_data['audio'])
                base_name = os.path.splitext(audio_filename)[0]
                output_file = f"{base_name}_q{i}_result.json"
                output_path = Path(output_folder) / output_file
                
                # Skip if already processed
                if output_path.exists():
                    continue
                
                result = self.process_question(video_file, question_data)
                
                # Save result
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                processed_count += 1
                
                # Rate limiting
                if processed_count < total_questions:
                    time.sleep(3)
    
    def process_task2_videos(self, video_folder: str, output_folder: str = "task2_results"):
        """Process Task 2: confused_video_m2 with fixed descriptive question"""
        os.makedirs(output_folder, exist_ok=True)
        
        video_folder_path = Path(video_folder)
        mp4_files = list(video_folder_path.glob("*.mp4"))
        
        for i, video_file in enumerate(mp4_files, 1):
            # Generate output filename
            output_file = f"{video_file.stem}_description_result.json"
            output_path = Path(output_folder) / output_file
            
            # Skip if already processed
            if output_path.exists():
                continue
            
            result = self.process_video_description(str(video_file))
            
            # Save result
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Rate limiting
            if i < len(mp4_files):
                time.sleep(3)

def main():
    """Main function for confused video benchmark evaluation"""
    # Configuration
    API_KEY = ""
    BASE_URL = ""
    
    processor = VideoProcessor(API_KEY, BASE_URL)
    
    # Task 1: Process confused_video with specific questions
    processor.process_task1_videos(
        video_folder="confused_video",
        json_path="avconfusebench_test_m1.json",
        output_folder="task1_results"
    )
    
    # Task 2: Process confused_video_m2 with fixed descriptive question
    processor.process_task2_videos(
        video_folder="confused_video_m2",
        output_folder="task2_results"
    )

if __name__ == "__main__":
    main() 