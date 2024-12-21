import pyaudio
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

        self.pyaudio: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(self.byte_width),
            channels=1,
            rate=self.sample_rate,
            output=True)

    def run(self) -> None:
        """
        Runs the application
        """

        pass

    def play_samples(self, samples: np.ndarray) -> None:
        """
        Play samples
        :param samples: array of sample
        """

        self.stream.write(
            (np.vectorize(clamp)(samples)).tobytes())

    def generate_frequency_sweep(self, start_freq: float, end_freq: float, duration: float) -> np.ndarray:
        """
        Generates a frequency sweep
        :param start_freq: starting frequency
        :param end_freq: frequency goal
        :param duration: duration
        :return: sample array
        """


def main():
    app = Application(sample_rate=48000)
    app.run()


if __name__ == '__main__':
    main()
