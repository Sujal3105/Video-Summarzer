"""
Video Transcription & Summarization App - STREAMLIT CLOUD DEPLOYMENT VERSION
Simplified UI: Just enter link + select language → Get summary
- API Key stored in Streamlit secrets (not shown to user)
- Transcription happens silently in backend
- Frontend: Link input + Language dropdown + Summary display
"""

import os
import re
import json
import shutil
import tempfile
import subprocess as sp
import glob
from pathlib import Path

import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from openai import OpenAI

from sarvamai import SarvamAI

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Video Summarizer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "current_summary" not in st.session_state:
    st.session_state.current_summary = None
if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None

# ============================================================================
# SECRETS MANAGEMENT
# ============================================================================

@st.cache_resource
def get_sarvam_key():
    """Get Sarvam API key from Streamlit secrets"""
    try:
        key = st.secrets.get("SARVAM_API_KEY")
        if not key:
            st.error("❌ API Key not configured. Contact admin.")
            st.stop()
        return key
    except Exception as e:
        st.error(f"❌ Error loading API key: {str(e)}")
        st.stop()

# ============================================================================
# AUDIO EXTRACTION (BACKEND)
# ============================================================================

def extract_audio(source: str, tmp_dir: str):
    """Extract audio from YouTube URL or local file"""
    out_wav = os.path.join(tmp_dir, "audio.wav")
    
    if re.match(r"^https?://", source.strip()):
        # YouTube download
        import yt_dlp
        
        format_strategies = [
            "worst[ext=mp4]",
            "worst",
            "bestaudio/best",
        ]
        
        client_options = ["web", "android", "ios"]
        info, last_err = None, None

        for strategy_idx, format_str in enumerate(format_strategies):
            for client in client_options:
                # Clean temp directory
                for f in os.listdir(tmp_dir):
                    fp = os.path.join(tmp_dir, f)
                    try:
                        if os.path.isfile(fp):
                            os.remove(fp)
                        elif os.path.isdir(fp):
                            shutil.rmtree(fp)
                    except:
                        pass

                opts = {
                    "format": format_str,
                    "outtmpl": os.path.join(tmp_dir, "%(title)s.%(ext)s"),
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                        "preferredquality": "192",
                    }],
                    "quiet": True,
                    "no_warnings": True,
                    "noplaylist": True,
                    "socket_timeout": 30,
                    "retries": 5,
                    "fragment_retries": 5,
                    "skip_unavailable_fragments": True,
                    "extractor_args": {
                        "youtube": {
                            "player_client": [client, "web"],
                            "max_retries": 5,
                        }
                    },
                    "http_headers": {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                }
                
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(source, download=True)
                    break
                except Exception as e:
                    last_err = str(e)
                    continue
            
            if info is not None:
                break

        if info is None:
            return None, None

        title = info.get("title", "video")
        wavs = [f for f in os.listdir(tmp_dir) if f.endswith(".wav")]
        
        if not wavs:
            audio_files = [f for f in os.listdir(tmp_dir) 
                          if f.endswith(('.m4a', '.webm', '.mp4', '.mp3', '.aac', '.opus'))]
            
            if not audio_files:
                return None, None
            
            input_file = os.path.join(tmp_dir, audio_files[0])
            
            r = sp.run(
                ["ffmpeg", "-y", "-i", input_file,
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_wav],
                capture_output=True, text=True, timeout=300
            )
            
            if r.returncode != 0:
                return None, None
        else:
            out_wav = os.path.join(tmp_dir, wavs[0])
        
        if not os.path.exists(out_wav):
            return None, None
        
        return out_wav, title

    else:
        # Local file
        source = os.path.expanduser(source)
        if not os.path.exists(source):
            return None, None
        
        title = Path(source).stem
        
        r = sp.run(
            ["ffmpeg", "-y", "-i", source,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_wav],
            capture_output=True, text=True, timeout=300
        )
        if r.returncode != 0:
            return None, None
        
        return out_wav, title

# ============================================================================
# AUDIO PREPROCESSING (BACKEND)
# ============================================================================

def preprocess_audio(wav_path: str, tmp_dir: str, denoise: bool = True, 
                    normalize: bool = True, highpass: bool = True, 
                    highpass_hz: int = 80, target_sr: int = 16000) -> str:
    """Preprocess audio"""
    
    audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)

    if highpass:
        from scipy.signal import butter, sosfilt
        sos = butter(4, highpass_hz, btype="highpass", fs=sr, output="sos")
        audio = sosfilt(sos, audio).astype(np.float32)

    if denoise:
        noise_len = int(0.5 * sr)
        noise_clip = audio[:noise_len] if len(audio) > noise_len else audio
        audio = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_clip,
            prop_decrease=0.8,
            stationary=True,
            n_fft=1024,
            n_jobs=-1,
        ).astype(np.float32)

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (10 ** (-1 / 20) / peak)

    clean_path = os.path.join(tmp_dir, "audio_clean.wav")
    sf.write(clean_path, audio, sr, subtype="PCM_16")
    
    return clean_path

# ============================================================================
# TRANSCRIPTION (BACKEND - NO UI)
# ============================================================================

def transcribe(audio_path: str, sarvam_key: str, model: str = "saaras:v3",
              mode: str = "transcribe", language: str = None) -> dict:
    """Transcribe audio using Sarvam AI (silent, no UI feedback)"""
    
    try:
        client = SarvamAI(api_subscription_key=sarvam_key)
        
        job = client.speech_to_text_job.create_job(
            model=model,
            mode=mode,
            language_code=language,
            with_diarization=False,
            num_speakers=None,
        )

        job.upload_files(file_paths=[audio_path])
        job.start()
        job.wait_until_complete()

        file_results = job.get_file_results()

        if file_results["failed"]:
            return None

        segments = []
        language_detected = "unknown"
        seg_id = 0

        with tempfile.TemporaryDirectory() as dl_dir:
            job.download_outputs(output_dir=dl_dir)

            json_files = glob.glob(os.path.join(dl_dir, "**", "*.json"), recursive=True)
            
            for jf in sorted(json_files):
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                language_detected = data.get("language_code", language_detected)

                # Handle timestamps
                ts = data.get("timestamps")
                if ts and isinstance(ts, dict):
                    texts = ts.get("words", []) or []
                    starts = ts.get("start_time_seconds", []) or []
                    ends = ts.get("end_time_seconds", []) or []
                    for txt, s, e in zip(texts, starts, ends):
                        if str(txt).strip():
                            segments.append({
                                "id": seg_id,
                                "start": float(s),
                                "end": float(e),
                                "text": str(txt).strip(),
                            })
                            seg_id += 1

                # Fallback to plain transcript
                if not segments:
                    full = data.get("transcript", "") or ""
                    if isinstance(full, list):
                        full = " ".join(str(t) for t in full)
                    full = str(full).strip()
                    if full:
                        segments.append({
                            "id": 0,
                            "start": 0.0,
                            "end": 0.0,
                            "text": full,
                        })

        if not segments:
            return None

        full_text = " ".join(s["text"] for s in segments)

        return {
            "language": language_detected,
            "text": full_text,
            "segments": segments
        }

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None

# ============================================================================
# SUMMARIZATION
# ============================================================================

def smart_chunk_text(text, max_chunk_tokens=3000, overlap=200):
    """Smart chunking that preserves sentence boundaries"""
    chars_per_token = 4
    max_chunk_chars = max_chunk_tokens * chars_per_token
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + " " + sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'```json|```', '', cleaned)
    cleaned = '\n'.join(line.rstrip() for line in cleaned.split('\n') if line.strip())
    return cleaned.strip()

def generate_summary_by_language(text: str, sarvam_key: str, language: str = "English") -> str:
    """Generate summary in selected language"""
    try:
        client = OpenAI(
            api_key=sarvam_key,
            base_url="https://api.sarvam.ai/v1"
        )
    except Exception as e:
        return None

    chunks = smart_chunk_text(text, max_chunk_tokens=3000, overlap=200)
    
    partials = []

    # Process chunks
    for i, chunk in enumerate(chunks):
        prompt = f"""
Generate a structured summary in {language}.
Summarize the following text in a clean and well-structured format.

Rules:
- Keep it medium length (not too long, not too short)
- Use clear headings and bullet points
- Explain key concepts simply
- Avoid unnecessary details and repetition
- Keep it concise but readable

Format:
1. Title
2. Short introduction (2-3 lines)
3. Key concepts (bullet points)
4. Example (if applicable)
5. Advantages
6. Limitations
7. Conclusion (2-3 lines)

Text:
{chunk}
"""

        try:
            response = client.chat.completions.create(
                model="sarvam-m",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )

            text_content = response.choices[0].message.content
            text_content = remove_think_tags(text_content)
            partials.append(text_content)
                
        except Exception as e:
            continue

    if not partials:
        return None

    combined = "\n\n".join(partials)

    # Final summary
    final_prompt = f"""
You are a professional technical writer.

Convert the following text into a CLEAN, WELL-STRUCTURED Markdown summary in {language}.

STRICT FORMATTING RULES:
- Use Markdown headings:
  # Title
  ## Short Introduction
  ## Key Concepts
  ## Example
  ## Advantages
  ## Limitations
  ## Conclusion
- Each section MUST be on a new line
- DO NOT merge sections into one paragraph
- Use bullet points (-) ONLY for Key Concepts, Advantages, and Limitations
- Keep sentences short and readable
- Add proper spacing between sections
- Format equations like: `y = mx + b`
- DO NOT write labels like "Title:" — use proper headings instead

CONTENT RULES:
- Keep total length between 150–250 words
- Avoid repetition
- Keep it clear and beginner-friendly

OUTPUT RULE:
- Return ONLY Markdown (no explanations, no extra text)

Text:
{combined}
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="sarvam-m",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful summarizer. Create clean, structured, medium-length summaries in {language}. Do NOT include <think> tags."
                    },
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            summary_text = response.choices[0].message.content.strip()
            summary_text = remove_think_tags(summary_text)
            
            if len(summary_text) > 100:
                return summary_text
                
        except Exception as e:
            continue

    return None

# ============================================================================
# UI - MAIN APP (SIMPLIFIED)
# ============================================================================

def main():
    """Main Streamlit application - Simplified UI"""
    
    # Get API key from secrets
    sarvam_key = get_sarvam_key()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🎬 Video Summarizer</h1>
        <p style="color: #666; font-size: 0.95rem;">
            Paste a YouTube link, select language, and get an instant summary
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Main form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_url = st.text_input(
            "🔗 Video Link",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed"
        )
    
    with col2:
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Marathi", "Tamil", "Telugu"],
            label_visibility="collapsed"
        )
    
    st.divider()
    
    # Process button
    if st.button("✨ Generate Summary", use_container_width=True, type="primary"):
        if not video_url:
            st.error("❌ Please enter a video link")
        else:
            with st.spinner("Processing... This may take 2-5 minutes"):
                try:
                    # Progress indicators
                    progress_placeholder = st.empty()
                    
                    progress_placeholder.info("📥 Extracting audio from video...")
                    
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Extract audio
                        audio_path, title = extract_audio(video_url, tmp_dir)
                        
                        if not audio_path:
                            st.error("❌ Could not extract audio. Check if the link is valid.")
                            st.stop()
                        
                        st.session_state.video_title = title
                        
                        # Preprocess
                        progress_placeholder.info("🎵 Preparing audio...")
                        clean_path = preprocess_audio(audio_path, tmp_dir)
                        
                        # Transcribe
                        progress_placeholder.info("📝 Transcribing video...")
                        result = transcribe(clean_path, sarvam_key)
                        
                        if not result:
                            st.error("❌ Transcription failed. Please try again.")
                            st.stop()
                        
                        st.session_state.transcription = result
                        
                        # Summarize
                        progress_placeholder.info(f"✍️ Generating {language} summary...")
                        summary = generate_summary_by_language(result["text"], sarvam_key, language)
                        
                        if not summary:
                            st.error("❌ Summary generation failed. Please try again.")
                            st.stop()
                        
                        st.session_state.current_summary = summary
                        progress_placeholder.success("✅ Summary ready!")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display summary if available
    if st.session_state.current_summary:
        st.divider()
        
        st.subheader(f"📖 {language} Summary")
        st.markdown(st.session_state.current_summary)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="⬇️ Download Summary",
                data=st.session_state.current_summary,
                file_name=f"summary_{st.session_state.video_title}_{language}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 New Summary", use_container_width=True):
                st.session_state.current_summary = None
                st.rerun()
    
    # Footer
    st.divider()
    st.caption("💡 Powered by Sarvam AI & OpenAI | Made with Streamlit")

if __name__ == "__main__":
    main()
