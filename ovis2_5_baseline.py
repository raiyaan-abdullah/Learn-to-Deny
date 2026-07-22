import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import AutoModelForCausalLM
import re
import pdb

# --- Constants ---
MODEL_PATH = "AIDC-AI/Ovis2.5-9B"
max_num_frames = 12
max_pixels = 896 * 896  # As per Ovis2.5 recommendation
max_new_tokens = 3072
thinking_budget = 2048
enable_thinking = False
enable_thinking_budget = False

avoid_videos = ["aggressive__talking_1_trim_0.mp4"]

# --- Load model ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

def sample_video_frames(video_path, num_frames=max_num_frames):
    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            stride = total_frames / num_frames
            indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
        frames = [Image.fromarray(clip.get_frame(i / clip.fps)) for i in indices]
        return frames

def split_think_and_answer(text):
    """
    Splits out the <think>…</think> section from the rest of the text.

    Returns a tuple (think_part, rest).
    If no <think> tag is found, think_part will be None and rest will be the original text.
    """
    # Use DOTALL so that '.' matches newlines
    pattern = re.compile(r'(<think>.*?</think>)', re.DOTALL)
    match = pattern.search(text)
    if match:
        think_part = match.group(1)
        # Remove the matched think block
        rest = text[:match.start()] + text[match.end():]
        return think_part, rest.strip()
    else:
        return None, text

def run_ovis_mcq(root_path, input_json_file, output_file):
    print(f"Evaluating videos in root path: {root_path}")
    # ---------- 1. Load data ----------
    with open(input_json_file, "r", encoding="utf-8") as f:
        video_data = json.load(f)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = []

    for entry in tqdm(video_data, desc=f"Evaluating {input_json_file}", total=len(video_data)):
        video_name = entry["video_name"]
        if video_name in avoid_videos:
            print(f"Skipping {video_name} as it's in the avoid list.")
            continue
        video_path = os.path.join(root_path, video_name)
        choices = entry["choices"]
        choices_block = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])

        # Compose the MCQ prompt
        question = (
            "The video shows a person or a group of people performing an action. From the list of options below, select the caption that most accurately describes what is happening in the video. There is also a choice if none of the other captions apply. Review the video and captions carefully. Only one choice is correct. Reply with the choice number only.\n"
            + choices_block
        )

        try:
            # Sample video frames
            frames = sample_video_frames(video_path, num_frames=max_num_frames)

            # Prepare messages in Ovis2.5 format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": question}
                ]
            }]

            # Preprocess inputs
            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                max_pixels=max_pixels,
                enable_thinking=enable_thinking
            )

            # Move inputs to GPU
            input_ids = input_ids.cuda()
            pixel_values = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
            grid_thws = grid_thws.cuda() if grid_thws is not None else None

            # Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    enable_thinking=enable_thinking,
                    enable_thinking_budget=enable_thinking_budget,
                    thinking_budget=thinking_budget,
                    eos_token_id=model.text_tokenizer.eos_token_id,
                    pad_token_id=model.text_tokenizer.pad_token_id
                )[0]
                response = model.text_tokenizer.decode(output_ids, skip_special_tokens=True)
                thinking, answer = split_think_and_answer(response) 

        except Exception as e:
            print(f"Ovis2.5 failed on {video_name}: {e}")
            continue

        none_index = entry["none_index"] + 1
        main_positive_index = entry["main_positive_index"] + 1 if "main_positive_index" in entry else None
        distractor_index = entry["distractor_index"] + 1 if "distractor_index" in entry else None

        results.append({
            "video_name": video_name,
            "question": question,
            "choices": choices_block,
            "model_response": answer,
            "thinking_path": thinking if enable_thinking else None,
            "none_index": none_index,
            "main_positive_index": main_positive_index,
            "distractor_index": distractor_index
        })

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        #pdb.set_trace()
    print(f"Results saved to {output_file}")