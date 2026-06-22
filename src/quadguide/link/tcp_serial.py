"""TCP serial-port emulator — replaces the UART with a TCP socket (HIL).

Drop-in replacement for link/serial_port.py:SerialPort. Implements the same
async interface (open / read_stream / write / close / is_connected) so the link
worker only chooses between the two by config (platform.serial.mode), with no
change to the RX/TX loops. MAVLink2 bytes flow over the socket exactly as they
would over the wire; the dev machine's ArduPilot SITL is the peer.

Disconnect handling mirrors SerialPort: read_stream raises ConnectionError, so
the worker reports DEGRADED health and runs its reconnect loop identically for
both transports.
"""
from __future__ import annotations

import asyncio
import socket
from typing import AsyncGenerator


class TCPSerialPort:
    def __init__(self, host: str, port: int):
        self._host      = host
        self._port      = port
        self._sock: socket.socket | None = None
        self._connected = False

    async def open(self) -> None:
        loop = asyncio.get_running_loop()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setblocking(False)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            await loop.sock_connect(self._sock, (self._host, self._port))
        except (ConnectionRefusedError, OSError) as exc:
            self._connected = False
            raise ConnectionError(
                f"TCPSerialPort: could not connect to {self._host}:{self._port}: {exc}"
            ) from exc
        self._connected = True

    async def read_stream(self) -> AsyncGenerator[int, None]:
        loop = asyncio.get_running_loop()
        while self._connected:
            try:
                data = await loop.sock_recv(self._sock, 64)
            except (ConnectionResetError, OSError) as exc:
                self._connected = False
                raise ConnectionError(str(exc)) from exc
            if not data:                       # peer closed the connection
                self._connected = False
                raise ConnectionError("TCPSerialPort: connection closed by peer")
            for b in data:
                yield b

    async def write(self, data: bytes) -> None:
        if not self._connected or self._sock is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.sock_sendall(self._sock, data)
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._connected = False

    def close(self) -> None:
        self._connected = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def is_connected(self) -> bool:
        return self._connected
