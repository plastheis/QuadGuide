from __future__ import annotations
import asyncio
from typing import AsyncGenerator

import serial


class SerialPort:
    def __init__(self, port: str, baud: int):
        self._port      = port
        self._baud      = baud
        self._ser: serial.Serial | None = None
        self._connected = False

    async def open(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            self._ser = await loop.run_in_executor(
                None, lambda: serial.Serial(self._port, self._baud, timeout=0.05)
            )
        except serial.SerialException as exc:
            self._connected = False
            raise ConnectionError(str(exc)) from exc
        self._connected = True

    async def read_stream(self) -> AsyncGenerator[int, None]:
        loop = asyncio.get_running_loop()
        while self._connected:
            try:
                data = await loop.run_in_executor(None, self._ser.read, 64)
            except serial.SerialException as exc:
                self._connected = False
                raise ConnectionError(str(exc)) from exc
            for b in data:
                yield b

    async def write(self, data: bytes) -> None:
        if not self._connected or self._ser is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._ser.write, data)
        except serial.SerialException:
            self._connected = False

    def close(self) -> None:
        self._connected = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self._connected
