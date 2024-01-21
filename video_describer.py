from openai import OpenAI
import requests
import base64
from moviepy.editor import VideoFileClip
import os
from gtts import gTTS
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

def extract_frames(video_path, frames_dir='frames', every_n_seconds=2):
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.mkdir(frames_dir)
    clip = VideoFileClip(video_path)
    duration = clip.duration  # Duration of the video in seconds
    
    # Extract frames at the specified interval
    for i in range(0, int(duration), every_n_seconds):
        frame = clip.get_frame(i)
        frame_path = f"{frames_dir}/frame_at_{i}_seconds.jpg"
        clip.save_frame(frame_path, i)  # Save the current frame as an image file

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def frame_difference(prev_frame, curr_frame, threshold=75000):
    """Calculate the difference between two frames."""
    #converting to rgb
    prev_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    curr_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

    #converting to grayscale
    gray_prev = cv2.cvtColor(prev_rgb, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_rgb, cv2.COLOR_BGR2GRAY)
    
    blur_prev = cv2.GaussianBlur(gray_prev, (21, 21), 0)
    blur_curr = cv2.GaussianBlur(gray_curr, (21, 21), 0)
    
    # Compute the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(blur_curr, blur_prev)
    
    # Threshold the difference image to binarize it, highlighting significant changes
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes, making the detected regions more complete
    dilated = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours in the thresholded image to identify regions of change
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the total area of all detected changes
    total_area = sum(cv2.contourArea(contour) for contour in contours)
    
    # Check if the total area of changes exceeds the threshold
    return total_area > threshold

def process_video(frames):
    key_frames = [frames[0]]
    for i in range(len(frames) - 1):
       if(frame_difference(cv2.imread(frames[i]), cv2.imread(frames[i+1]))):
          key_frames.append(frames[i+1])

    return key_frames
  
def submit_request(base64_image):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
  }
  
  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe what is in this image, be as consise as possible pointing out main features. Stick to this all the time."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }
  
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  data = response.json()
  return data['choices'][0]['message']['content']

def summarise(descriptions):
  prompt = "Summarize, into one paragraph, the following key moments from a video, be as consise as possible only picking out key features: \n" + "\n".join(descriptions)
  response = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": "You are a helpful summarising assistant, be as concise as possible so to only point out things that occur a lot in the images."},
        {"role": "user", "content": prompt}
      ],
      max_tokens=150,
      temperature=0.7,
  )
  print("Summarised the video")
  return response.choices[0].message.content

def text_to_speech(text):
  print("Converting text to speech")
  tts = gTTS(text=text, lang='en')
  tts.save("summary.mp3")
  os.system("afplay summary.mp3")

def run(video_path):
  extract_frames(video_path)

  # Assume we have a directory 'frames' with extracted images
  frames_dir = 'frames'
  frames = []
  for frame_filename in os.listdir(frames_dir):
    frame_path = os.path.join(frames_dir, frame_filename)
    frames.append(frame_path)
  key_frames = process_video(frames=frames)
  descriptions = []

  descriptions = execute_requests_concurrently(key_frames, descriptions)
  text_to_speech(summarise(descriptions=descriptions))

def execute_requests_concurrently(images, descriptions):
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all the API requests
        future_to_prompt = [executor.submit(submit_request, encode_image(image)) for image in images]
        
        # Process the results as they are completed
        for future in as_completed(future_to_prompt):
            try:
                result = future.result()
                descriptions.append(result)
            except Exception as exc:
                print(f"Prompt generated an exception: {future.exception}\n{exc}\n")
        return descriptions

# def select_video():
#     """Open a dialog to select a video file."""
#     file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
#     if file_path:
#         entry.delete(0, tk.END)
#         entry.insert(0, file_path)

# def summarize_video():
#     """Placeholder function for video summarization logic."""
#     video_path = entry.get()
#     if video_path:
#         # Implement your video summarization logic here
#         messagebox.showinfo("Info", f"Video summarization started for: {video_path}")
#         run(video_path=video_path)
#     else:
#         messagebox.showwarning("Warning", "Please select a video file first.")

# # Create the main window
# root = tk.Tk()
# root.title("Video Summarizer")

# # Create and pack widgets
# entry = tk.Entry(root, width=50)
# entry.pack(padx=10, pady=10)

# select_button = tk.Button(root, text="Select Video", command=select_video)
# select_button.pack(pady=5)

# summarize_button = tk.Button(root, text="Summarize Video", command=summarize_video)
# summarize_button.pack(pady=5)

# # Run the application
# root.mainloop()

video_path = "IMG_4491.mov"
run(video_path)