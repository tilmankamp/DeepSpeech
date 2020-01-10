import os
import io
import sox
import wave
import opuslib
import tempfile
import collections
from webrtcvad import Vad

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = (DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)

AUDIO_TYPE_WAV = 'audio/wav'
AUDIO_TYPE_OPUS = 'audio/opus'


def write_audio_format_to_wav_file(wav_file, audio_format=DEFAULT_FORMAT):
    rate, channels, width = audio_format
    wav_file.setframerate(rate)
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(width)


def read_audio_format_from_wav_file(wav_file):
    return wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()


def get_num_samples(buffer_len, audio_format=DEFAULT_FORMAT):
    _, channels, width = audio_format
    return buffer_len // (channels * width)


def get_pcm_duration(pcm_len, audio_format=DEFAULT_FORMAT):
    return get_num_samples(pcm_len, audio_format) / audio_format[0]


def convert_audio(src_audio_path, dst_audio_path, file_type=None, audio_format=DEFAULT_FORMAT):
    sample_rate, channels, width = audio_format
    transformer = sox.Transformer()
    transformer.set_output_format(file_type=file_type, rate=sample_rate, channels=channels, bits=width*8)
    transformer.build(src_audio_path, dst_audio_path)


class AudioFile:
    def __init__(self, audio_path, as_path=False, audio_format=DEFAULT_FORMAT):
        self.audio_path = audio_path
        self.audio_format = audio_format
        self.as_path = as_path
        self.open_file = None
        self.tmp_file_path = None

    def __enter__(self):
        if self.audio_path.endswith('.wav'):
            self.open_file = wave.open(self.audio_path, 'r')
            if read_audio_format_from_wav_file(self.open_file) == self.audio_format:
                if self.as_path:
                    self.open_file.close()
                    return self.audio_path
                return self.open_file
            self.open_file.close()
        _, self.tmp_file_path = tempfile.mkstemp(suffix='.wav')
        convert_audio(self.audio_path, self.tmp_file_path, file_type='wav', audio_format=self.audio_format)
        if self.as_path:
            return self.tmp_file_path
        self.open_file = wave.open(self.tmp_file_path, 'r')
        return self.open_file

    def __exit__(self, *args):
        if not self.as_path:
            self.open_file.close()
        if self.tmp_file_path is not None:
            os.remove(self.tmp_file_path)


def read_frames(wav_file, frame_duration_ms=30, yield_remainder=False):
    audio_format = read_audio_format_from_wav_file(wav_file)
    frame_size = int(audio_format[0] * (frame_duration_ms / 1000.0))
    while True:
        try:
            data = wav_file.readframes(frame_size)
            if not yield_remainder and get_pcm_duration(len(data), audio_format) * 1000 < frame_duration_ms:
                break
            yield data
        except EOFError:
            break


def read_frames_from_file(audio_path, audio_format=DEFAULT_FORMAT, frame_duration_ms=30, yield_remainder=False):
    with AudioFile(audio_path, audio_format=audio_format) as wav_file:
        for frame in read_frames(wav_file, frame_duration_ms=frame_duration_ms, yield_remainder=yield_remainder):
            yield frame


def vad_split(audio_frames,
              audio_format=DEFAULT_FORMAT,
              num_padding_frames=10,
              threshold=0.5,
              aggressiveness=3):
    sample_rate, channels, width = audio_format
    if channels != 1:
        raise ValueError('VAD-splitting requires mono samples')
    if width != 2:
        raise ValueError('VAD-splitting requires 16 bit samples')
    if sample_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError('VAD-splitting only supported for sample rates 8000, 16000, 32000, or 48000')
    if aggressiveness not in [0, 1, 2, 3]:
        raise ValueError('VAD-splitting aggressiveness mode has to be one of 0, 1, 2, or 3')
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    vad = Vad(int(aggressiveness))
    voiced_frames = []
    frame_duration_ms = 0
    frame_index = 0
    for frame_index, frame in enumerate(audio_frames):
        frame_duration_ms = get_pcm_duration(len(frame), audio_format) * 1000
        if int(frame_duration_ms) not in [10, 20, 30]:
            raise ValueError('VAD-splitting only supported for frame durations 10, 20, or 30 ms')
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > threshold * ring_buffer.maxlen:
                triggered = False
                yield b''.join(voiced_frames), \
                      frame_duration_ms * max(0, frame_index - len(voiced_frames)), \
                      frame_duration_ms * frame_index
                ring_buffer.clear()
                voiced_frames = []
    if len(voiced_frames) > 0:
        yield b''.join(voiced_frames), \
              frame_duration_ms * (frame_index - len(voiced_frames)), \
              frame_duration_ms * (frame_index + 1)


def pack_number(n, num_bytes):
    return n.to_bytes(num_bytes, 'big', signed=False)


def unpack_number(data):
    return int.from_bytes(data, 'big', signed=False)


def write_opus(opus_file, audio_format, audio_data):
    rate, channels, width = audio_format
    frame_size = 60 * rate // 1000
    encoder = opuslib.Encoder(rate, channels, opuslib.APPLICATION_AUDIO)
    chunk_size = frame_size * channels * width
    opus_file.write(pack_number(len(audio_data), 4))
    opus_file.write(pack_number(rate, 2))
    opus_file.write(pack_number(channels, 1))
    opus_file.write(pack_number(width, 1))
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        encoded = encoder.encode(chunk, frame_size)
        opus_file.write(pack_number(len(encoded), 2))
        opus_file.write(encoded)


def read_opus_header(opus_file):
    opus_file.seek(0)
    pcm_len = unpack_number(opus_file.read(4))
    rate = unpack_number(opus_file.read(2))
    channels = unpack_number(opus_file.read(1))
    width = unpack_number(opus_file.read(1))
    return pcm_len, (rate, channels, width)


def read_opus(opus_file):
    pcm_len, audio_format = read_opus_header(opus_file)
    rate, channels, _ = audio_format
    frame_size = 60 * rate // 1000
    decoder = opuslib.Decoder(rate, channels)
    audio_data = bytearray()
    while len(audio_data) < pcm_len:
        chunk_size = unpack_number(opus_file.read(2))
        chunk = opus_file.read(chunk_size)
        decoded = decoder.decode(chunk, frame_size)
        audio_data.extend(decoded)
    audio_data = audio_data[:pcm_len]
    return audio_format, audio_data


def read_wav(wav_data):
    with io.BytesIO(wav_data) as base_wav_file:
        with wave.open(base_wav_file, 'rb') as wav_file:
            return read_audio_format_from_wav_file(wav_file), wav_file.readframes()


def read_audio(audio_type, audio_file):
    if audio_type == AUDIO_TYPE_WAV:
        return read_wav(audio_file)
    elif audio_type == AUDIO_TYPE_OPUS:
        return read_opus(audio_file)
    else:
        raise ValueError('Unsupported audio format: {}'.format(audio_type))


def read_wav_duration(wav_file):
    with wave.open(wav_file, 'rb') as wav_file_reader:
        return wav_file_reader.getnframes() / wav_file_reader.getframerate()


def read_opus_duration(opus_file):
    pcm_len, audio_format = read_opus_header(opus_file)
    return get_pcm_duration(pcm_len, audio_format)


def read_duration(audio_type, audio_file):
    if audio_type == AUDIO_TYPE_WAV:
        return read_wav_duration(audio_file)
    elif audio_type == AUDIO_TYPE_OPUS:
        return read_opus_duration(audio_file)
    else:
        raise ValueError('Unsupported audio format: {}'.format(audio_type))


def convert_to_wav(audio_type, audio_file):
    audio_format = None
    if audio_type == AUDIO_TYPE_WAV:
        return audio_file
    else:
        audio_format, audio_data = read_audio(audio_type, audio_file)
    memory_wav_file = io.BytesIO()
    with wave.open(memory_wav_file, 'wb') as wav_file:
        write_audio_format_to_wav_file(wav_file, audio_format)
        wav_file.writeframes(audio_data)
    memory_wav_file.seek(0)
    return memory_wav_file
