import os
import random
import re
import textwrap
from typing import List, TypedDict

from dotenv import load_dotenv
from PIL import Image, ImageDraw
from moviepy.editor import concatenate_videoclips, ImageClip, AudioFileClip
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

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
    "machine learning fundamentals",
    "natural language processing (NLP)",
    "computer vision",
    "AI ethics",
    "generative AI",
    "large language models (LLMs)",
    "AI in healthcare",
    "AI in finance",
    "autonomous systems",
    "AI and creativity",
    "AI tools and platforms",
    "AI career paths",
    "AI model evaluation",
    "prompt engineering",
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
    prompt = PromptTemplate.from_template(
        "Write a ~300-word tutorial on the topic: {topic}. "
        "Do not include any code or implementation details. "
        "The tutorial should be a clear and engaging general overview suitable for a beginner-level audience. "
        "Avoid technical jargon and focus on explaining the concept and its significance."
    )
    tutorial = llm.predict(prompt.format(topic=topic))
    return {"tutorial": tutorial}


# ========== STEP 3: Generate Per-Paragraph Audio ==========
def text_to_audio(state):
    import shutil
    from elevenlabs import ElevenLabs, save, Voice, VoiceSettings
    import pyttsx3

    tutorial = state["tutorial"]
    paragraphs = [p.strip() for p in tutorial.split("\n") if p.strip()]
    audio_paths = []

    if os.path.exists("audio"):
        shutil.rmtree("audio")
    os.makedirs("audio", exist_ok=True)

    for i, paragraph in enumerate(paragraphs):
        audio_path = f"audio/narration_{i:02}.mp3"

        if USE_ELEVENLABS and ELEVENLABS_API_KEY:
            try:
                client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
                audio = client.generate(
                    text=paragraph,
                    voice=Voice(
                        voice_id="rwRSRQs2Lguppd3kDP6p",  # Rachel
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
                print(f"⚠️ ElevenLabs error on paragraph {i}, falling back to pyttsx3:", e)
                engine = pyttsx3.init()
                engine.save_to_file(paragraph, audio_path)
                engine.runAndWait()
        else:
            engine = pyttsx3.init()
            engine.save_to_file(paragraph, audio_path)
            engine.runAndWait()

        audio_paths.append(audio_path)

    return {"paragraphs": paragraphs, "audio_paths": audio_paths}

# ========== STEP 4: Create Slides ==========
def create_slides(state):
    import shutil
    if os.path.exists("slides"):
        shutil.rmtree("slides")
    os.makedirs("slides", exist_ok=True)

    paragraphs = state["paragraphs"]
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

# ========== STEP 5: Create Synced Video ==========
def create_video(state):
    topic = state["topic"]
    safe_topic = sanitize_filename(topic)

    base_dir = "Videos"
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, safe_topic)
    os.makedirs(output_dir, exist_ok=True)

    slide_paths = state["slide_paths"]
    audio_paths = state["audio_paths"]

    clips = []
    for i, (slide, audio) in enumerate(zip(slide_paths, audio_paths)):
        audio_clip = AudioFileClip(audio)
        duration = audio_clip.duration
        img_clip = ImageClip(slide).set_duration(duration).set_audio(audio_clip)
        clips.append(img_clip)

    final = concatenate_videoclips(clips)
    video_path = os.path.join(output_dir, f"{safe_topic}_tutorial.mp4")
    final.write_videofile(video_path, fps=1)

    return {
        "video_path": video_path,
        "slide_paths": slide_paths,
        "audio_path": audio_paths
    }

# ========== LangGraph Setup ==========
class TutorialState(TypedDict, total=False):
    topic: str
    tutorial: str
    paragraphs: List[str]
    slide_paths: List[str]
    audio_paths: List[str]
    video_path: str

builder = StateGraph(TutorialState)
builder.add_node("generate_topic", generate_topic)
builder.add_node("generate_tutorial", generate_tutorial)
builder.add_node("text_to_audio", text_to_audio)
builder.add_node("create_slides", create_slides)
builder.add_node("create_video", create_video)

builder.set_entry_point("generate_topic")
builder.add_edge("generate_topic", "generate_tutorial")
builder.add_edge("generate_tutorial", "text_to_audio")
builder.add_edge("text_to_audio", "create_slides")
builder.add_edge("create_slides", "create_video")
builder.add_edge("create_video", END)

graph = builder.compile()

# ========== Run the Flow ==========
if __name__ == "__main__":
    final_state = graph.invoke({})
    print("\n🎉 All done! Your synced video tutorial has been created.")
