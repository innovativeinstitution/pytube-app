import os
import random
import re
from typing import List, TypedDict

from dotenv import load_dotenv
import pyttsx3
import textwrap
from PIL import Image, ImageDraw
from moviepy.editor import ImageSequenceClip, AudioFileClip
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_ELEVENLABS = os.getenv("USE_ELEVENLABS", "false").lower() == "true"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.3,
    max_tokens=1000,
    openai_api_key=OPENAI_API_KEY
)

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name[:50]

# ========== STEP 1: Generate Topic ==========
categories = [
    "web development", "algorithms", "clean code", "APIs", "cloud computing",
    "DevOps", "testing", "design patterns", "CI/CD", "security",
    "performance optimization", "refactoring", "mobile apps", "containers",
]

def generate_topic(state):
    category = random.choice(categories)
    prompt = PromptTemplate.from_template(
        "Give me a unique and creative software engineering topic in the area of {category} that would make a good 2-minute tutorial."
    )
    topic = llm.predict(prompt.format(category=category))
    return {"topic": topic}

# ========== STEP 2: Generate Tutorial ==========
def generate_tutorial(state):
    topic = state["topic"]
    prompt = PromptTemplate.from_template("Write a ~300-word tutorial on the topic: {topic} but don't include hashtags or other characters that would make TTS difficult or awkward.")
    tutorial = llm.predict(prompt.format(topic=topic))
    return {"tutorial": tutorial}

# ========== STEP 3: Convert to Audio ==========
def text_to_audio(state):
    tutorial = state["tutorial"]
    audio_path = "narration.mp3"

    if USE_ELEVENLABS and ELEVENLABS_API_KEY:
        try:
            from elevenlabs import ElevenLabs, save, Voice, VoiceSettings

            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

            audio = client.generate(
                text=tutorial,
                voice=Voice(
                    voice_id="EXAVITQu4vr4xnSDxMaL",  # Rachel
                    settings=VoiceSettings(
                        stability=0.3,
                        similarity_boost=0.85,
                        style=0.7,
                        use_speaker_boost=True
                    )
                ),
                model="eleven_multilingual_v2"
            )

            save(audio, audio_path)
        except Exception as e:
            print("⚠️ ElevenLabs error, falling back to pyttsx3:", e)
            engine = pyttsx3.init()
            engine.save_to_file(tutorial, audio_path)
            engine.runAndWait()
    else:
        engine = pyttsx3.init()
        engine.save_to_file(tutorial, audio_path)
        engine.runAndWait()

    return {"audio_path": audio_path}

# ========== STEP 4: Create Slides ==========
def create_slides(state):
    if os.path.exists("slides"):
        import shutil
        shutil.rmtree("slides")
    os.makedirs("slides", exist_ok=True)

    tutorial = state["tutorial"]
    paragraphs = tutorial.split("\n")
    slide_paths = []

    for i, paragraph in enumerate(paragraphs):
        img = Image.new("RGB", (1280, 720), color="white")
        draw = ImageDraw.Draw(img)
        wrapped = textwrap.fill(paragraph, width=80)
        draw.text((50, 100), wrapped, fill="black")
        slide_path = f"slides/slide_{i:02}.png"
        img.save(slide_path)
        slide_paths.append(slide_path)

    return {"slide_paths": slide_paths}

# ========== STEP 5: Create Video ==========
def create_video(state):
    topic = state["topic"]
    safe_topic = sanitize_filename(topic)

    base_dir = "Videos"
    os.makedirs(base_dir, exist_ok=True)

    output_dir = os.path.join(base_dir, safe_topic)
    os.makedirs(output_dir, exist_ok=True)

    # Move slides there
    slide_paths = []
    for i, slide_path in enumerate(state["slide_paths"]):
        new_path = os.path.join(output_dir, f"slide_{i:02}.png")
        os.rename(slide_path, new_path)
        slide_paths.append(new_path)

    # Move audio there
    audio_src = state["audio_path"]
    audio_dst = os.path.join(output_dir, "narration.mp3")
    os.rename(audio_src, audio_dst)

    # Create video
    clip = ImageSequenceClip(slide_paths, fps=1)
    audio = AudioFileClip(audio_dst)
    final = clip.set_audio(audio).set_duration(audio.duration)

    video_filename = f"{safe_topic}_tutorial.mp4"
    video_path = os.path.join(output_dir, video_filename)
    final.write_videofile(video_path, fps=1)

    return {
        "video_path": video_path,
        "slide_paths": slide_paths,
        "audio_path": audio_dst
    }

# ========== STEP 6: Upload to YouTube (Optional) ==========
def upload_to_youtube(state):
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
    credentials = flow.run_local_server(port=8080, prompt='consent', authorization_prompt_message='')

    youtube = build("youtube", "v3", credentials=credentials)

    video_path = state["video_path"]
    topic = state["topic"]

    request_body = {
        "snippet": {
            "title": f"Quick Tutorial: {topic}",
            "description": f"This is a 2-minute tutorial on {topic}. Auto-generated using AI!",
            "tags": ["AI", "tutorial", "software engineering", topic],
            "categoryId": "28"
        },
        "status": {
            "privacyStatus": "unlisted"
        }
    }

    media = MediaFileUpload(video_path)
    response = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=media
    ).execute()

    print(f"✅ Video uploaded: https://youtu.be/{response['id']}")
    return {"youtube_url": f"https://youtu.be/{response['id']}"}

# ========== LangGraph Setup ==========
class TutorialState(TypedDict, total=False):
    topic: str
    tutorial: str
    audio_path: str
    slide_paths: List[str]
    video_path: str
    youtube_url: str

builder = StateGraph(TutorialState)

builder.add_node("generate_topic", generate_topic)
builder.add_node("generate_tutorial", generate_tutorial)
builder.add_node("text_to_audio", text_to_audio)
builder.add_node("create_slides", create_slides)
builder.add_node("create_video", create_video)
# builder.add_node("upload_to_youtube", upload_to_youtube)  # Enable if needed

builder.set_entry_point("generate_topic")
builder.add_edge("generate_topic", "generate_tutorial")
builder.add_edge("generate_tutorial", "text_to_audio")
builder.add_edge("text_to_audio", "create_slides")
builder.add_edge("create_slides", "create_video")
builder.add_edge("create_video", END)
# builder.add_edge("create_video", "upload_to_youtube")
# builder.add_edge("upload_to_youtube", END)

graph = builder.compile()

# ========== Run the Flow ==========
if __name__ == "__main__":
    final_state = graph.invoke({})
    print("\n🎉 All done! Your tutorial has been created.")
