"""
Azure TTS Module — optimized.
Reuses the synthesizer instance to avoid re-auth on every call.
"""

import os
import azure.cognitiveservices.speech as speechsdk

# ─── Persistent synthesizer (avoids re-handshake every call) ─────────────────

_speech_config = speechsdk.SpeechConfig(
    subscription=os.environ.get("AZURE_SPEECH_KEY", "YOUR_AZURE_SPEECH_KEY_HERE"),
    region=os.environ.get("AZURE_SPEECH_REGION", "eastus")
)
_speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
# Request compressed audio for faster network transfer
_speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
)


def build_ssml(text: str) -> str:
    """Build SSML with whispering style and slightly faster rate."""
    # Escape XML special chars
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<speak version="1.0"
       xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts"
       xml:lang="en-US">
  <voice name="en-US-AriaNeural">
    <mstts:express-as style="whispering">
      <prosody rate="0.95" pitch="-0.4st">
        {text}
      </prosody>
    </mstts:express-as>
  </voice>
</speak>"""


def speak(text: str, output_file: str = "tts.wav") -> str:
    """
    Synthesize text to a WAV file using Azure Neural TTS.
    Returns the output file path.
    """
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=_speech_config,
        audio_config=audio_config
    )

    result = synthesizer.speak_ssml_async(build_ssml(text)).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return output_file
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Azure TTS failed: {details.error_details}")
    else:
        raise RuntimeError("Azure TTS failed (unknown reason)")
