import streamlit as st
import  streamlit_toggle as tog
from multiprocessing import Process
import asyncio
import sounddevice
import time
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from apscheduler.schedulers.background import BackgroundScheduler


save_transcript = []
st.set_page_config('Real Time Speech Analytics')
st.header('Real Time Speech to Text')
placeholder = st.empty()
toggle_state = tog.st_toggle_switch(label="Toggle to Speak")


class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
              for alt in result.alternatives:
                  save_transcript.append(alt.transcript+'\n')
                  with open('./data/transcript.txt', 'w') as f:
                      f.writelines(save_transcript)
                  print(alt.transcript)


async def mic_stream():
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def write_chunks(stream):
    async for chunk, status in mic_stream():
        with open('./data/message.txt', 'r') as f:
          message = f.read()
          if message == 'start':
              await stream.input_stream.send_audio_event(audio_chunk=chunk)
          else:
              break
    await stream.input_stream.end_stream()


async def basic_transcribe():
    client = TranscribeStreamingClient(region="us-east-2")
    try:
      stream = await client.start_stream_transcription(
          language_code="en-US",
          media_sample_rate_hz=16000,
          media_encoding="pcm"
      )
    except Exception as e:
        print(e)
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events())


def read_file():
    with placeholder:
        while True:
          with open('./data/transcript.txt', 'r') as f:
              st.markdown(''.join(f.readlines()))
              time.sleep(1)


def trigger_speech():
    asyncio.run(basic_transcribe())

scheduler = BackgroundScheduler()
scheduler.add_job(func=trigger_speech)
scheduler.start()

if toggle_state:
    with open('./data/message.txt', 'w') as f:
        f.write('start')
else:
    with open('./data/message.txt', 'w') as f:
        f.write('stop')


p1 = Process(read_file())
p1.start()
