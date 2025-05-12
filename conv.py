# This file is hardcoded to transparently reproduce HEARME_zh.wav
# Therefore it may NOT generalize gracefully to other texts
# Refer to Usage in README.md for more general usage patterns

# pip install kokoro>=0.8.1 "misaki[zh]>=0.8.1"
from kokoro import KModel, KPipeline
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import tqdm
import librosa

import re
from podcastfy.client import generate_podcast
conversation_config = {
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
    "user_instructions": "成品稿件中不可出現可識別的人名，如果有真實人名出現，請修正為代稱，例如小華老師、阿明等",
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

# The original texts variable is removed, as we will use the dialogue from the file.

en_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID, model=False)


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
    sf.write(filename, wav, SAMPLE_RATE)
    return wav


def convert_to_mp3(wav_file, mp3_file, sample_rate=SAMPLE_RATE):
    """Converts a WAV file to MP3 using librosa and soundfile."""
    try:
        # Load the WAV file using librosa
        y, sr = librosa.load(wav_file, sr=sample_rate)

        # Save as MP3 using soundfile with the correct subtype
        sf.write(mp3_file, y, sr, format='MP3', subtype='PCM_16')  # Or another suitable subtype

        print(f"Successfully converted {wav_file} to {mp3_file}")

    except Exception as e:
        print(f"Error converting {wav_file} to MP3: {e}")


if __name__ == "__main__":

    filepath = generate_podcast(
        urls=["https://www.boyo.org.tw/boyoV2/special_column/teaching_site/%E8%80%81%E5%B8%AB%E4%BD%A0%E6%98%8E%E5%B9%B4%E9%82%84%E6%9C%83%E6%95%99%E6%88%91%E5%80%91%E5%97%8E%EF%BC%9F"],
        transcript_only=True,
        topic="老師你明年還會教我們嗎？",
        conversation_config=conversation_config,
        llm_model_name="gemini-2.0-flash-lite"
    )

    dialogue_list = read_and_split_dialogue(filepath)

    combined_audio = []
    for i, (speaker, dialogue) in enumerate(tqdm.tqdm(dialogue_list, desc="Generating speech")):
        if speaker == "person1":
            voice = PERSON1_VOICE
        elif speaker == "person2":
            voice = PERSON2_VOICE
        else:
            print(f"警告：未知的說話者：{speaker}")
            continue

        wav_filename = path / f"dialogue_{i:03}_{speaker}.wav"  # 每個對話都儲存為獨立檔案
        wav = generate_speech(dialogue, voice, wav_filename)
        combined_audio.append(wav)

        # Convert each individual WAV file to MP3
        mp3_filename = path / f"dialogue_{i:03}_{speaker}.mp3"
        convert_to_mp3(wav_filename, mp3_filename)


    combined_audio = np.concatenate(combined_audio)
    combined_wav_filename = path / 'combined_dialogue.wav'
    sf.write(combined_wav_filename, combined_audio, SAMPLE_RATE)

    # Convert the combined WAV file to MP3
    combined_mp3_filename = path / 'combined_dialogue.mp3'
    convert_to_mp3(combined_wav_filename, combined_mp3_filename)


    print("Speech generation complete. Audio saved to combined_dialogue.wav/mp3 and individual files.")
