import os
import random
import re
import shutil
import textwrap
import time
from typing import List, TypedDict

from dotenv import load_dotenv
from PIL import Image, ImageDraw
from moviepy.editor import concatenate_videoclips, ImageClip, AudioFileClip
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from moviepy.editor import VideoFileClip
from langchain.llms import Ollama


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_ELEVENLABS = os.getenv("USE_ELEVENLABS", "false").lower() == "true"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Environment-driven LLM selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

if LLM_PROVIDER == "ollama":
    print("🤖 Using local Ollama model:", OLLAMA_MODEL)
    llm = Ollama(model=OLLAMA_MODEL)
else:
    print("🌐 Using OpenAI Chat model: gpt-4o-mini")
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

def clean_llm_output(text: str) -> str:
    # Remove <think>...</think> blocks entirely
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Optional: remove standalone <think> or </think> if they still exist
    text = text.replace("<think>", "").replace("</think>", "")

    return text.strip()

def generate_topic(state):
    print("➡️  Step: Generate Topic")
    start = time.time()

    category = random.choice(categories)
    prompt = PromptTemplate.from_template(
        "Give me a unique and creative software engineering topic in the area of {category} that would make a good 2-minute tutorial."
    )
    
    if LLM_PROVIDER == "ollama":
        raw = llm.predict(prompt.format(category=category))
        topic = clean_llm_output(raw)
    else:
        topic = llm.predict(prompt.format(category=category))

    end = time.time()
    print(f"✅ Done: Generate Topic in {end - start:.2f}s")
    return {"topic": topic}

# ========== STEP 2: Generate Tutorial ==========
def generate_tutorial(state):
    print("➡️  Step: Generate Tutorial")
    start = time.time()

    topic = state["topic"]
    prompt = PromptTemplate.from_template(
        "Write a ~300-word tutorial on the topic: {topic}. "
        "Avoid using code blocks, symbols like #, <, *, or any special characters that could confuse text-to-speech. "
        "Keep the tutorial clear, natural, and spoken-word friendly. "
        "No implementation details or technical formatting — just a clean, engaging overview."
    )
    
    if LLM_PROVIDER == "ollama":
        raw = llm.predict(prompt.format(topic=topic))
        tutorial = clean_llm_output(raw)
    else:
        tutorial = llm.predict(prompt.format(topic=topic))

    end = time.time()
    print(f"✅ Done: Generate Tutorial in {end - start:.2f}s")
    return {"tutorial": tutorial}

# ========== STEP 3: Generate Per-Paragraph Audio ==========
def text_to_audio(state):
    print("➡️  Step: Text to Audio")
    start = time.time()

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
                        voice_id="rwRSRQs2Lguppd3kDP6p",
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

    end = time.time()
    print(f"✅ Done: Text to Audio in {end - start:.2f}s")
    return {"paragraphs": paragraphs, "audio_paths": audio_paths}

# ========== STEP 4: Create Slides ==========
def create_slides(state):
    print("➡️  Step: Create Slides")
    start = time.time()

    if os.path.exists("slides"):
        shutil.rmtree("slides")
    os.makedirs("slides", exist_ok=True)

    from PIL import ImageFont

    paragraphs = state["paragraphs"]
    slide_paths = []

    for i, paragraph in enumerate(paragraphs):
        img = Image.new("RGB", (1280, 720), color="#0f172a")
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, 1280, 20], fill="#2563eb")
        draw.rectangle([0, 700, 1280, 720], fill="#1e3a8a")

        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()

        wrapped = textwrap.fill(paragraph, width=60)
        draw.text((60, 80), wrapped, fill="white", font=font)

        slide_path = f"slides/slide_{i:02}.png"
        img.save(slide_path)
        slide_paths.append(slide_path)

    end = time.time()
    print(f"✅ Done: Create Slides in {end - start:.2f}s")
    return {"slide_paths": slide_paths}

# ========== STEP 5: Create Synced Video ==========
def create_video(state):
    import subprocess

    print("➡️  Step: Create Video")
    start = time.time()

    topic = state["topic"]
    safe_topic = sanitize_filename(topic)

    base_dir = "Videos"
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, safe_topic)
    os.makedirs(output_dir, exist_ok=True)

    slide_paths = state["slide_paths"]
    audio_paths = state["audio_paths"]

    # Step 1: Create generated tutorial video
    clips = []
    for i, (slide, audio) in enumerate(zip(slide_paths, audio_paths)):
        audio_clip = AudioFileClip(audio)
        duration = audio_clip.duration
        img_clip = ImageClip(slide).set_duration(duration).set_audio(audio_clip)
        clips.append(img_clip)

    raw_generated_path = os.path.join(output_dir, f"{safe_topic}_tutorial_raw.mp4")
    final = concatenate_videoclips(clips)
    final.write_videofile(raw_generated_path, fps=1)

    # Step 2: Normalize both videos for compatibility
    def normalize_video(input_path, output_path, width=1280, height=720):
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"scale={width}:{height},fps=30",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ], check=True)

    intro_path = "Intro.mp4"
    assert os.path.exists(intro_path), f"❌ Missing intro video: {intro_path}"
    assert os.path.exists(raw_generated_path), f"❌ Missing generated video: {raw_generated_path}"

    normalized_intro = os.path.join(output_dir, "intro_normalized.mp4")
    normalized_generated = os.path.join(output_dir, "generated_normalized.mp4")

    print("🎬 Normalizing videos for concat...")
    normalize_video(intro_path, normalized_intro)
    normalize_video(raw_generated_path, normalized_generated)

    # Step 3: Create concat list
    concat_list_path = os.path.join(output_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        f.write(f"file '{os.path.abspath(normalized_intro).replace(os.sep, '/')}'\n")
        f.write(f"file '{os.path.abspath(normalized_generated).replace(os.sep, '/')}'\n")

    print("🔗 FFmpeg concat list:")
    with open(concat_list_path) as f:
        print(f.read())

    # Step 4: Final merge
    final_video_path = os.path.join(output_dir, f"{safe_topic}_tutorial.mp4")
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        final_video_path
    ], check=True)

    # Step 5: Clean up
    if os.path.exists("audio"):
        shutil.rmtree("audio")
    if os.path.exists("slides"):
        shutil.rmtree("slides")

    end = time.time()
    print(f"✅ Done: Create Video in {end - start:.2f}s")
    return {
        "video_path": final_video_path,
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
    print("🚀 Starting tutorial generation...\n")
    start_time = time.time()

    final_state = graph.invoke({})

    end_time = time.time()
    print("\n🎉 All done! Your synced video tutorial has been created.")
    print(f"🕒 Total time: {end_time - start_time:.2f} seconds")
