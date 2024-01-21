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

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

def extract_frames(video_path, frames_dir='frames', every_n_seconds=1):
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
  
def submit_request(base64_image):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
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
        {"role": "system", "content": "You are a helpful summarising assistant."},
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
  descriptions = []
  for frame_filename in os.listdir(frames_dir):
      frame_path = os.path.join(frames_dir, frame_filename)
      encoded_frame = encode_image(frame_path)
      descriptions.append(submit_request(encoded_frame))

  text_to_speech(summarise(descriptions=descriptions))

def select_video():
    """Open a dialog to select a video file."""
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def summarize_video():
    """Placeholder function for video summarization logic."""
    video_path = entry.get()
    if video_path:
        # Implement your video summarization logic here
        messagebox.showinfo("Info", f"Video summarization started for: {video_path}")
        run(video_path=video_path)
    else:
        messagebox.showwarning("Warning", "Please select a video file first.")

# Create the main window
root = tk.Tk()
root.title("Video Summarizer")

# Create and pack widgets
entry = tk.Entry(root, width=50)
entry.pack(padx=10, pady=10)

select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack(pady=5)

summarize_button = tk.Button(root, text="Summarize Video", command=summarize_video)
summarize_button.pack(pady=5)

# Run the application
root.mainloop()