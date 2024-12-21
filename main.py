import pyaudio
import numpy as np


class Application:
    """
    Main application class
    """

    def __init__(self, sample_rate: int, byte_width: int):
        self.sample_rate: int = sample_rate
        self.byte_width: int = byte_width

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


def main():
    app = Application(
        sample_rate=48000,
        byte_width=4)
    app.run()


if __name__ == '__main__':
    main()
