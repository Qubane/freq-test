import pyaudio
import asyncio
import numpy as np


def clamp(x: float, min_: float = -1, max_: float = 1) -> float:
    """
    Clamps the value to be in range of -1 to 1
    :param x: value
    :param min_: minimum allowed
    :param max_: maximum allowed
    :return: clamped value
    """

    return max(min_, min(max_, x))


class Application:
    """
    Main application class
    """

    def __init__(self, sample_rate: int):
        self.sample_rate: int = sample_rate
        self.byte_width: int = 4
        self.chunk_size: int = 1024

        self.pyaudio: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(self.byte_width),
            channels=1,
            rate=self.sample_rate,
            input=True, output=True)

    def run(self) -> None:
        """
        Runs the application
        """

        async def coro():
            play_samples = self.generate_frequency_sweep(20, 20000, 2)
            result = await asyncio.gather(
                self.play_samples(play_samples),
                self.record_samples(len(play_samples) / self.sample_rate))
            record_samples = result[1]
            with open('good.txt', "w", encoding="utf-8") as file:
                for val in record_samples:
                    file.write(f"{val}\n")

        asyncio.run(coro())

    async def play_samples(self, samples: np.ndarray) -> None:
        """
        Play samples
        :param samples: array of sample
        """

        data = (np.vectorize(clamp)(samples)).tobytes()
        for i in range(0, len(data), self.chunk_size):
            await self.play_chunk(data[i:i + self.chunk_size])

    async def record_samples(self, duration: float) -> np.ndarray:
        """
        Records samples from the microphone
        :param duration: record duration
        :return: array of samples
        """

        samples = np.zeros(int(self.sample_rate * duration), dtype=np.float32)
        for i in range(0, len(samples), self.chunk_size):
            chunk = await self.record_chunk()
            if i + self.chunk_size >= len(samples):
                chunk = chunk[:len(samples) - (len(samples) // self.chunk_size * self.chunk_size)]
            samples[i:i+self.chunk_size] = chunk
        return samples

    async def play_chunk(self, chunk: bytes) -> None:
        """
        Plays an audio chunk.
        :param chunk: chunk of audio
        """

        self.stream.write(chunk)

    async def record_chunk(self) -> np.ndarray:
        """
        Records a chunk of audio from default microphone
        :return: an array of samples
        """

        raw_data = self.stream.read(self.chunk_size)
        return np.frombuffer(raw_data, dtype=np.float32)

    def generate_frequency_sweep(self, start_freq: float, end_freq: float, duration: float) -> np.ndarray:
        """
        Generates a frequency sweep
        :param start_freq: starting frequency
        :param end_freq: frequency goal
        :param duration: duration
        :return: sample array
        """

        samples = np.zeros(int(self.sample_rate * duration), dtype=np.float32)

        t = 0
        freq_step = start_freq / self.sample_rate * np.pi * 2
        freq_change_rate = (end_freq - start_freq) / self.sample_rate**2 * np.pi * 2 / duration
        for i in range(len(samples)):
            samples[i] = np.sin(t)
            t += freq_step
            freq_step += freq_change_rate

        return samples


def main():
    app = Application(sample_rate=48000)
    app.run()


if __name__ == '__main__':
    main()
