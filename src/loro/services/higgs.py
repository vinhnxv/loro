"""Client for the Higgs Audio v3 TTS server (sglang-omni).

The server only accepts reference audio as a server-local path or an HTTP URL,
so the reference voice is served over a temporary HTTP server that the TTS
host fetches from (works across Tailscale). Adapted from
speech/higgs-audio/client/higgs_tts_vi.py, but keeps one HTTP server alive for
the whole batch of segments instead of one per call.
"""

import functools
import os
import socket
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from loro.config import Config
from loro.harness.retry import with_retry


def _local_ip_towards(host_url: str) -> str:
    server_host = host_url.split("//", 1)[-1].split(":")[0].split("/")[0]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((server_host, 80))
        return s.getsockname()[0]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class HiggsClient:
    """Synthesizes speech cloned from one reference voice. Use as a context manager."""

    def __init__(self, cfg: Config, ref_audio: Path, ref_text: str):
        self.cfg = cfg
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self._server: ThreadingHTTPServer | None = None
        self._ref_url = ""

    def __enter__(self):
        handler = functools.partial(_QuietHandler, directory=str(self.ref_audio.parent))
        self._server = ThreadingHTTPServer(("0.0.0.0", 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        ip = _local_ip_towards(self.cfg.higgs_host)
        port = self._server.server_address[1]
        self._ref_url = f"http://{ip}:{port}/{self.ref_audio.name}"
        return self

    def __exit__(self, *exc):
        if self._server:
            self._server.shutdown()

    def synthesize(self, text: str, output: Path, voice: str | None = None) -> None:
        # `voice` is part of the shared synthesize surface for the preset engine;
        # Higgs clones a reference voice and ignores it (KTD1).
        def call() -> None:
            response = requests.post(
                f"{self.cfg.higgs_host}/v1/audio/speech",
                json={
                    "model": self.cfg.higgs_model,
                    "input": text,
                    "response_format": "wav",
                    "references": [{"audio_path": self._ref_url, "text": self.ref_text}],
                },
                timeout=self.cfg.higgs_timeout,
            )
            response.raise_for_status()
            tmp = output.with_name(f".tmp.{output.name}.{uuid.uuid4().hex}")
            tmp.write_bytes(response.content)
            os.replace(tmp, output)

        with_retry("tts", call, attempts=self.cfg.retry_attempts,
                   base_delay=self.cfg.retry_base_delay)
