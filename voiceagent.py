import asyncio
import logging
import os
import sys
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
)
from livekit.plugins.elevenlabs import VoiceSettings

chat_history = []

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from livekit.plugins import deepgram, elevenlabs
from bot import generate_response

# Load env
load_dotenv()
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
eleven_api_key = os.getenv("ELEVENLABS_API_KEY")

logger = logging.getLogger("transcribe")
logging.basicConfig(level=logging.DEBUG)

# === Warm-up OpenAI ===
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

warm_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
warm_llm.invoke([HumanMessage(content="Hi!")])


def fetch_response(user_input):
    logger.debug(f"[fetch_response] User input: {user_input}")
    chat_history.append({"role": "user", "content": user_input})
    result = generate_response(chat_history, voice_mode=True)
    chat_history.append({"role": "assistant", "content": result["text"]})
    logger.debug(f"[fetch_response] Response from bot: {result}")
    return result["text"], result.get("language", "hi"), result.get("image_urls", [])


async def entrypoint(ctx: JobContext):
    logger.info(f"🚀 Starting transcriber for room: {ctx.room.name}")

    # === Audio Setup ===
    audio_src = rtc.AudioSource(sample_rate=44100, num_channels=1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot-tts", audio_src)

    # === Deepgram STT ===
    stt_impl = deepgram.STT(
        model="nova-3-general",
        api_key=deepgram_api_key,
        language="multi",
        interim_results=True,
        punctuate=True,
        no_delay=True,
        filler_words=True,
        profanity_filter=True,
        numerals=True,
    )

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        logger.info(f"🎧 Starting transcription for: {participant.identity}")
        audio_stream = rtc.AudioStream(track)
        stt_stream = stt_impl.stream()

        async def _handle_audio_stream():
            async for ev in audio_stream:
                stt_stream.push_frame(ev.frame)

        async def _handle_transcription_output():
            async for ev in stt_stream:
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    user_query = ev.alternatives[0].text.strip()
                    if not user_query:
                        logger.warning("⚠️ Empty user query, skipping.")
                        continue

                    logger.info(f"📝 User query: {participant.identity}: {user_query}")

                    async def respond_and_speak():
                        response_text, lang, image_urls = await asyncio.to_thread(
                            fetch_response, user_query
                        )
                        logger.info(f"💬 Bot response [{lang}]: {response_text}")
                        if image_urls:
                            for url in image_urls:
                                logger.info(f"image url: {url}")

                        tts = elevenlabs.TTS(
                            voice_id="VJzrUxHaC52mTyYHMCnK",
                            model="eleven_turbo_v2_5",
                            encoding="mp3_44100_128",
                            api_key=eleven_api_key,
                            language=lang,
                            oice_settings=VoiceSettings(speed=0.8),
                        )

                        synth_stream = tts.synthesize(response_text)
                        async for chunk in synth_stream:
                            await audio_src.capture_frame(chunk.frame)

                    asyncio.create_task(respond_and_speak())

        await asyncio.gather(
            _handle_audio_stream(),
            _handle_transcription_output(),
        )

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(transcribe_track(participant, track))

    @ctx.room.on("participant_joined")
    def on_participant_joined(participant):
        logger.info(f"👤 Participant joined: {participant.identity}")

    @ctx.room.on("track_published")
    def on_track_published(publication, participant):
        logger.info(f"🛁 Track published from {participant.identity}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.room.local_participant.publish_track(audio_track)
    logger.info("✅ Ready and listening...")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
