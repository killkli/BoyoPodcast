import gradio as gr
from kokoro import KModel, KPipeline
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import re
from pydub import AudioSegment
from podcastfy.client import generate_podcast

# Default values for conversation configuration
DEFAULT_CONVERSATION_CONFIG = {
    "conversation_style": [
        "有趣幽默",
        "節奏明快",
        "活潑互動"
    ],
    "roles_person1": "內容整理達人",
    "roles_person2": "好奇發問者",
    "dialogue_structure": [
        "開場破冰",
        "重點精華",
        "歡樂收尾"
    ],
    "podcast_name": "博幼進行式",
    "podcast_tagline": "把教育和希望帶到偏鄉",
    "output_language": "Traditional Chinese",
    "engagement_techniques": [
        "反問句引導",
        "生活小故事",
        "生動比喻",
        "機智幽默"
    ],
    "creativity": 1,
    "user_instructions": "成品稿件中不可出現可識別的人名，如果有真實人名出現，請修正為代稱，例如小華老師、阿明等，請重複檢查文本中有無人名，有的話務必改用代稱",
    "max_num_chunks": 8,
    "min_chunk_size": 600,
    "text_to_speech": {
        "output_directories": {
            "transcripts": "./data/transcripts",
            "audio": "./data/audio"
        },
        "ending_message": "我們下次見！"
    }
}

DEFAULT_COURSE_CONVERSATION_CONFIG = {
    "conversation_style": [
        "循循善誘",
        "條理清晰",
        "深入淺出"
    ],
    "roles_person1": "資深講師",
    "roles_person2": "求知慾強烈的學生",
    "dialogue_structure": [
        "課程目標介紹",
        "概念講解",
        "範例演示",
        "重點總結",
        "課後練習"
    ],
    "podcast_name": "AI學習之旅",
    "podcast_tagline": "輕鬆掌握AI知識",
    "output_language": "Traditional Chinese",
    "engagement_techniques": [
        "提問互動",
        "類比生活情境",
        "拆解複雜概念",
        "鼓勵發問"
    ],
    "creativity": 0.7,  # 適度創意，但保持教學的嚴謹性
    "user_instructions": "請根據提供的課程資料，以對話的形式講解課程內容。Person1負責講解，Person2負責提問。確保講解清晰易懂，並提供實例輔助理解。避免過於艱澀的術語，並適時鼓勵學生提問。在講解完一個概念後，請進行簡單的總結。最後，提供一些課後練習，幫助學生鞏固所學知識。",
    "max_num_chunks": 10,  # 課程內容可能較長，增加chunk數量
    "min_chunk_size": 500,  # 稍微縮小chunk size，更精細的控制
    "text_to_speech": {
        "output_directories": {
            "transcripts": "./data/transcripts",
            "audio": "./data/audio"
        },
        "ending_message": "今天的課程就到這裡，下次見！"
    }
}


REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
SAMPLE_RATE = 24000

# How much silence to insert between paragraphs: 5000 is about 0.2 seconds
N_ZEROS = 5000

# Whether to join sentences in paragraphs 1 and 3
JOIN_SENTENCES = True

# Define voices for person1 and person2
PERSON1_VOICE = 'zf_021'  # Female voice
PERSON2_VOICE = 'zm_025'  # Male voice

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'

en_pipeline = KPipeline(lang_code='a', repo_id=REPO_ID, model=False)


def en_callable(text):
    if text == 'Kokoro':
        return 'kˈOkəɹO'
    elif text == 'Sol':
        return 'sˈOl'
    return next(en_pipeline(text)).phonemes

# HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
# Simple piecewise linear fn that decreases speed as len_ps increases


def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1


model = KModel(repo_id=REPO_ID).to(device).eval()
zh_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID,
                        model=model, en_callable=en_callable)

path = Path(__file__).parent


def generate_speech(dialogue, voice, filename):
    """Generates speech for a given dialogue and saves it to a file."""
    generator = zh_pipeline(dialogue, voice=voice, speed=speed_callable)
    result = next(generator)
    wav = result.audio
    return wav


def convert_to_mp3(wav_file, mp3_file, sample_rate=SAMPLE_RATE):
    """Converts a WAV file to MP3 using librosa and soundfile."""
    try:
        # Use pydub to convert WAV to MP3
        sound = AudioSegment.from_wav(wav_file)
        sound.export(mp3_file, format="mp3")

        print(f"Successfully converted {wav_file} to {mp3_file}")

    except Exception as e:
        print(f"Error converting {wav_file} to MP3: {e}")


def read_and_split_dialogue(filepath):
    """
    讀取包含 <Person1> 和 <Person2> 對話的 txt 檔案，並依照原始順序將對話內容和說話者資訊存到一個 list。

    Args:
        filepath (str): txt 檔案的路徑。

    Returns:
        list: 包含 (speaker, dialogue) tuple 的 list，其中 speaker 為 "person1" 或 "person2"。
               如果檔案不存在或格式不正確，則返回 []。
    """

    dialogue_list = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()  # 一次性讀取整個檔案內容

            # 使用正則表達式匹配 Person1 和 Person2 的對話
            matches = re.findall(
                r'<(Person1|Person2)>\s*"?(.*?)"?\s*</(Person1|Person2)>', content, re.DOTALL)

            for match in matches:
                speaker = match[0].lower()  # 將 Person1 或 Person2 轉換為小寫
                dialogue = match[1]
                dialogue_list.append((speaker, dialogue))

    except FileNotFoundError:
        print(f"錯誤：找不到檔案：{filepath}")
        return []
    except Exception as e:
        print(f"發生錯誤：{e}")
        return []

    return dialogue_list


def generate_podcast_and_speech(url, topic, conversation_style, roles_person1, roles_person2, dialogue_structure, podcast_name, podcast_tagline, output_language, engagement_techniques, creativity, user_instructions, max_num_chunks, min_chunk_size, person1_voice, person2_voice, text_to_speech_ending_message):
    """Generates the podcast, speech, and returns the audio file."""

    conversation_config = {
        "conversation_style": conversation_style,
        "roles_person1": roles_person1,
        "roles_person2": roles_person2,
        "dialogue_structure": dialogue_structure,
        "podcast_name": podcast_name,
        "podcast_tagline": podcast_tagline,
        "output_language": output_language,
        "engagement_techniques": engagement_techniques,
        "creativity": creativity,
        "user_instructions": user_instructions,
        "max_num_chunks": max_num_chunks,
        "min_chunk_size": min_chunk_size,
        "person1_voice": person1_voice,
        "person2_voice": person2_voice,
        "text_to_speech": {
            # Keep directories constant
            "output_directories": DEFAULT_CONVERSATION_CONFIG["text_to_speech"]["output_directories"],
            "ending_message": text_to_speech_ending_message  # Make ending message dynamic
        }
    }

    filepath = generate_podcast(
        urls=[url],
        transcript_only=True,
        topic=topic,
        conversation_config=conversation_config,
        llm_model_name="gemini-2.0-flash-lite"
    )

    dialogue_list = read_and_split_dialogue(filepath)

    combined_audio = []
    for i, (speaker, dialogue) in enumerate(dialogue_list):
        if speaker == "person1":
            voice = conversation_config.get("person1_voice", PERSON1_VOICE)
        elif speaker == "person2":
            voice = conversation_config.get("person2_voice", PERSON2_VOICE)
        else:
            print(f"警告：未知的說話者：{speaker}")
            continue

        wav_filename = path / f"dialogue_{i:03}_{speaker}.wav"  # 每個對話都儲存為獨立檔案
        wav = generate_speech(dialogue, voice, wav_filename)
        combined_audio.append(wav)

    combined_audio = np.concatenate(combined_audio)
    combined_wav_filename = path / 'combined_dialogue.wav'
    sf.write(combined_wav_filename, combined_audio, SAMPLE_RATE)

    # Convert the combined WAV file to MP3
    combined_mp3_filename = path / 'combined_dialogue.mp3'
    convert_to_mp3(combined_wav_filename, combined_mp3_filename)

    print("Speech generation complete. Audio saved to combined_dialogue.wav/mp3 and individual files.")

    return combined_mp3_filename


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Podcast and Speech Generation Tool")

        with gr.Row():
            url_input = gr.Textbox(
                label="Podcast URL", value="https://www.boyo.org.tw/boyoV2/special_column/boyo_stories/%E6%88%91%E5%8F%AA%E6%98%AF%E7%9A%AE%EF%BC%8C%E6%88%91%E4%B8%8D%E5%A3%9E%EF%BC%81")
            topic_input = gr.Textbox(label="Podcast Topic", value="我只是皮，我不壞！")

        config_choice = gr.Radio(
            choices=["Default", "Course"],
            label="Configuration Preset",
            value="Default"  # Default selection
        )

        with gr.Accordion("Conversation Configuration", open=False):
            conversation_style_input = gr.CheckboxGroup(
                choices=["有趣幽默", "節奏明快", "活潑互動", "循循善誘", "條理清晰", "深入淺出"],
                label="Conversation Style",
                value=DEFAULT_CONVERSATION_CONFIG["conversation_style"]
            )
            roles_person1_input = gr.Textbox(
                label="Role of Person 1", value=DEFAULT_CONVERSATION_CONFIG["roles_person1"])
            roles_person2_input = gr.Textbox(
                label="Role of Person 2", value=DEFAULT_CONVERSATION_CONFIG["roles_person2"])
            dialogue_structure_input = gr.CheckboxGroup(
                choices=["開場破冰", "重點精華", "歡樂收尾", "課程目標介紹",
                         "概念講解", "範例演示", "重點總結", "課後練習"],
                label="Dialogue Structure",
                value=DEFAULT_CONVERSATION_CONFIG["dialogue_structure"]
            )
            podcast_name_input = gr.Textbox(
                label="Podcast Name", value=DEFAULT_CONVERSATION_CONFIG["podcast_name"])
            podcast_tagline_input = gr.Textbox(
                label="Podcast Tagline", value=DEFAULT_CONVERSATION_CONFIG["podcast_tagline"])
            output_language_input = gr.Textbox(
                label="Output Language", value=DEFAULT_CONVERSATION_CONFIG["output_language"])
            engagement_techniques_input = gr.CheckboxGroup(
                choices=["反問句引導", "生活小故事", "生動比喻", "機智幽默", "提問互動",
                         "類比生活情境", "拆解複雜概念", "鼓勵發問"],  # Added course options
                label="Engagement Techniques",
                value=DEFAULT_CONVERSATION_CONFIG["engagement_techniques"]
            )
            creativity_input = gr.Slider(
                minimum=0, maximum=1, step=0.1, value=DEFAULT_CONVERSATION_CONFIG["creativity"], label="Creativity")
            user_instructions_input = gr.Textbox(
                label="User Instructions", value=DEFAULT_CONVERSATION_CONFIG["user_instructions"])
            max_num_chunks_input = gr.Number(
                label="Max Number of Chunks", value=DEFAULT_CONVERSATION_CONFIG["max_num_chunks"])
            min_chunk_size_input = gr.Number(
                label="Min Chunk Size", value=DEFAULT_CONVERSATION_CONFIG["min_chunk_size"])
            person1_voice_input = gr.Textbox(
                label="Person 1 Voice", value=PERSON1_VOICE)
            person2_voice_input = gr.Textbox(
                label="Person 2 Voice", value=PERSON2_VOICE)
            text_to_speech_ending_message_input = gr.Textbox(
                label="Ending Message", value=DEFAULT_CONVERSATION_CONFIG["text_to_speech"]["ending_message"])

        generate_button = gr.Button("Generate Podcast and Speech")

        audio_output = gr.Audio(label="Generated Audio")

        def update_config_values(config_choice):
            if config_choice == "Course":
                config = DEFAULT_COURSE_CONVERSATION_CONFIG
                engagement_value = config["engagement_techniques"]
                conversation_style_value = config["conversation_style"]
                dialogue_structure_value = config["dialogue_structure"]

            else:  # Default
                config = DEFAULT_CONVERSATION_CONFIG
                engagement_value = config["engagement_techniques"]
                conversation_style_value = config["conversation_style"]
                dialogue_structure_value = config["dialogue_structure"]

            return (
                conversation_style_value,
                roles_person1_input.value,
                roles_person2_input.value,
                dialogue_structure_value,
                podcast_name_input.value,
                podcast_tagline_input.value,
                output_language_input.value,
                engagement_value,
                creativity_input.value,
                user_instructions_input.value,
                max_num_chunks_input.value,
                min_chunk_size_input.value,
            )

        config_choice.change(
            fn=update_config_values,
            inputs=[config_choice],
            outputs=[
                conversation_style_input,
                roles_person1_input,
                roles_person2_input,
                dialogue_structure_input,
                podcast_name_input,
                podcast_tagline_input,
                output_language_input,
                engagement_techniques_input,
                creativity_input,
                user_instructions_input,
                max_num_chunks_input,
                min_chunk_size_input,
            ]
        )

        generate_button.click(
            fn=generate_podcast_and_speech,
            inputs=[
                url_input,
                topic_input,
                conversation_style_input,
                roles_person1_input,
                roles_person2_input,
                dialogue_structure_input,
                podcast_name_input,
                podcast_tagline_input,
                output_language_input,
                engagement_techniques_input,
                creativity_input,
                user_instructions_input,
                max_num_chunks_input,
                min_chunk_size_input,
                person1_voice_input,
                person2_voice_input,
                text_to_speech_ending_message_input
            ],
            outputs=[audio_output]
        )

    demo.launch()


if __name__ == "__main__":
    main()
