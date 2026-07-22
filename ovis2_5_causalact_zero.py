import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from moviepy.editor import VideoFileClip
from transformers import AutoModelForCausalLM
import re
import pdb
import time

# --- Constants ---
max_num_frames = 12
max_pixels = 896 * 896  # As per Ovis2.5 recommendation
max_new_tokens = 3072
thinking_budget = 2048
enable_thinking = False
enable_thinking_budget = False

avoid_videos = ["NotBabyCrawling/v_NotBabyCrawling_g06_c17.mp4", "NotBabyCrawling/v_NotBabyCrawling_g06_c19.mp4"]

# --- Question Templates Dictionary ---
QUESTION_TEMPLATES = {
    "data_and_graph": (
        r"""
        Your primary task is to act as a causal reasoning engine for video understanding. You will be given a set of observed components from a video. Your goal is to use the provided causal graph structure to logically determine the final action and select the most accurate descriptive caption from a list.

        ### Component Definitions
        
        First, understand the meaning of each component (node) in our causal graph. These are the building blocks we use to describe any event in a video.

        * P (Persons): This identifies the person or group of people present in the video. For example, 'a man', 'a woman', 'two children'.
        * O (Objects): This lists the main inanimate objects that are present in the video. For example, 'a guitar', 'a basketball', 'a chair'.
        * L (Location): This describes the environment or setting where the event takes place. For example, 'a park', 'a kitchen', 'an office'.
        * S (Spatial Relation): This describes the positioning of persons and objects relative to each other and the location. It's a snapshot of *where* things are. For example, 'a person is standing next to a table'.
        * I (Interaction): This describes the sustained physical contact or manipulation between a person and an object over a duration. The presence of an interaction is a crucial clue. For example, 'a person's hands are on a keyboard' or 'a child's feet are on the bicycle pedals'. If there is no contact, the interaction is 'None'.
        * M (Motion): This describes the specific movements and body motions of the persons. It is the dynamic part of the video. For example, 'fingers pressing keys', 'legs pedaling', 'arms swinging'.
        * A (Action): This is the high-level activity that you must infer. It is the final conclusion based on all the other components. For example, 'typing', 'biking', 'tennis swing'. There might be no relevant action happening at all.

        ---

        ### The Causal Graph: How Components are Related

        The components above are not independent. They influence each other in a specific order, which we can represent as a Directed Acyclic Graph (DAG). Think of it as a flowchart where one thing leads to another.

        The directed edges (shown with right arrows → or \u2192 character) show the direction of influence. For example, `P → I` means that the presence of a Person (P) is a prerequisite for an Interaction (I) to occur.

        The directed edges of the graph are:
        P → I
        O → I
        P → S
        O → S
        L → S
        I → M
        M → A
        I → A

        Here are the causal relationships (the directed edges) explained in detail:

        * `P → I` and `O → I` (How Interaction is formed): For an Interaction (I) to happen, there must be both a Person (P) to perform the interaction and an Object (O) to be interacted with. The interaction is the direct result of the person engaging with the object.
        * `P → S`, `O → S`, and `L → S` (How Spatial Relation is formed): The Spatial Relation (S) is determined by where the Persons (P) and Objects (O) are located within the Location (L).
        * `I → M` (Interaction causes Motion): The specific way a person Interacts (I) with an object over time is what causes the Motion (M) we see. For example, the interaction of 'hands on a guitar' may cause the motion of 'fingers strumming strings'.
        * `I → A` and `M → A` (How the final Action is determined): The final Action (A) is a direct result of both the Interaction (I) and the Motion (M). The Interaction tells you *what* is being engaged, and the Motion tells you *how* it is being engaged. Together, they define the action.

        ---

        ### Illustrative Examples

        To make this perfectly clear, let's walk through two examples.

        #### Example 1: Interaction is PRESENT
        Imagine a video where a person is playing a guitar.

        * P (Persons): 'a person'
        * O (Objects): 'a guitar'
        * L (Location): 'a room'
        * Causal Analysis:
            1.  The 'person' (P) and 'guitar' (O) exist in the 'room' (L).
            2.  Their Spatial Relation (S) is: 'The person is sitting and holding the guitar'.
            3.  Because the person is holding the guitar, a direct Interaction (I) occurs: 'The person's hands are on the guitar's neck and strings'.
            4.  This interaction may lead to a specific Motion (M): 'Fingers are pressing on the frets and the other hand is strumming the strings'.
            5.  Therefore, the combination of the Interaction (I) ('hands on guitar') and the Motion (M) ('strumming') leads to the undeniable conclusion for Action (A): 'playing guitar'.

        #### Example 2: Interaction is ABSENT
        Now, imagine a video of the same person and guitar, but the person is not playing it.

        * P (Persons): 'a person'
        * O (Objects): 'a guitar'
        * L (Location): 'a room'
        * Causal Analysis:
            1.  The 'person' (P) and 'guitar' (O) still exist in the 'room' (L).
            2.  However, the Spatial Relation (S) is different: 'The person is sitting on a chair, and the guitar is leaning against the wall next to them'.
            3.  Because the person is not touching the guitar, the Interaction (I) is: 'None'.
            4.  Since there is no interaction, no relevant Motion (M) is generated. The observed motion might be 'The person is sitting still'.
            5.  Without a direct Interaction (I) or a related Motion (M) involving the guitar, the Action (A) cannot be 'playing guitar'.

        ---

        ### Your Task

        You will now be given a new video. The goal is to select the single best caption from a list.

        Infer the components internally. Do not print them.
        * P (Persons): Think what person or group of people is present in the video.
        * O (Objects): Think what main inanimate objects are present in the video.
        * L (Location): Think about the environment or setting where the event takes place in.
        * S (Spatial Relation): Think about the positioning of persons and objects relative to each other and the location.
        * I (Interaction): Carefully observe and think about any sustained interaction (physical contact or manipulation) between a person and an object over a duration. There might be no interaction at all.
        * M (Motion): Carefully observe and think about the specific movements and body motions of the persons.
        * A (Action): ??? Unknown, you must infer this.

        Use the causal relationships (directed edges) and the component definitions from above to decide the Action. Reason with these components in your head. Do not output your reasoning.

        From the list of options below, select the caption that most accurately describes the action happening in the video.
        There is also a choice if none of the other captions apply.
        Review the video and captions carefully. Only one choice is correct.
        Reply with the choice number only.

        Output format: reply with the choice number only.
        """
    )
}

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

def run_ovis_mcq(model, root_path, input_json_file, output_file, question_template):
    print(f"Evaluating videos in root path: {root_path}")
    
    with open(input_json_file, "r", encoding="utf-8") as f:
        video_data = json.load(f)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = []

    for entry in tqdm(video_data, desc=f"Evaluating {input_json_file}", total=len(video_data)):
        video_name = entry["video_name"]
        if video_name in avoid_videos:
            print(f"Skipping {video_name} due to known issues.")
            continue
        video_path = os.path.join(root_path, video_name)
        choices = entry["choices"]
        choices_block = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])

        question = question_template + choices_block

        try:
            frames = sample_video_frames(video_path, num_frames=max_num_frames)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": question}
                ]
            }]

            input_ids, pixel_values, grid_thws = model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                max_pixels=max_pixels,
                enable_thinking=enable_thinking
            )

            input_ids = input_ids.cuda()
            pixel_values = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
            grid_thws = grid_thws.cuda() if grid_thws is not None else None

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

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Ovis MCQ evaluation with configurable questions")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output JSON file for results")
    parser.add_argument("--question_type", type=str, required=True,
                        choices=list(QUESTION_TEMPLATES.keys()),
                        help="Type of question template to use")
    return parser.parse_args()

if __name__ == "__main__":
    start_ts = time.time()
    print("Start system time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts)))
    
    args = parse_arguments()
    
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    
    root_path = '/home/ra164195/Datasets/UCF-101-neg/val'
    input_json = "/home/ra164195/investigate_ucf101/common_mcq_data/ucf_101_neg_val_mcq.json"
    
    question_template = QUESTION_TEMPLATES[args.question_type]
    
    print(f"Available question types: {list(QUESTION_TEMPLATES.keys())}")
    print(f"Selected question type: {args.question_type}")

    run_ovis_mcq(
        model,
        root_path,
        input_json,
        args.output_file,
        question_template
    )

    end_ts = time.time()
    print("End system time:  ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts)))
    print(f"Total elapsed time: {end_ts - start_ts:.2f} seconds")