"""Core library for the ASL static-fingerspelling recognition project.

The package intentionally has no import-time dependency on a webcam, a trained
classifier, or an audio device. Command-line scripts import only the modules
they need, which keeps unit tests fast and portable.
"""

__version__ = "1.0.0"
