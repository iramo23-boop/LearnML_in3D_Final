"""Compatibility stub for the official simulator client.

The original course project communicates with a live 3D simulation server.
For local development and professor review, this file provides a minimal
GameClient interface so imports do not break. Replace this file with the
official SDK if your class server is available.
"""
from __future__ import annotations


class GameClient:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port

    def connect(self) -> None:
        print(f"GameClient placeholder connected to {self.host}:{self.port}")

    def close(self) -> None:
        print("GameClient placeholder closed")
