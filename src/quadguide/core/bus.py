from __future__ import annotations
import fcntl
import multiprocessing
import os
import select
import struct
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

from quadguide.core.messages import (
    TrackerEstimate,  FMT_TRACKER_ESTIMATE,
    TargetEstimate,   FMT_TARGET_ESTIMATE,
    AttitudeState,    FMT_ATTITUDE_STATE,
    IMUFrame,         FMT_IMU_FRAME,
    AccelCmd,         FMT_ACCEL_CMD,
    ControlCmd,       FMT_CONTROL_CMD,
    LockOnCmd,        FMT_LOCKON_CMD,
    HealthReport,     FMT_HEALTH_REPORT,
)

__all__ = ["Bus", "TOPICS"]

TOPICS: dict[str, tuple[type, str]] = {
    "kcf/estimate":    (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "nano/estimate":   (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "target/estimate": (TargetEstimate,  FMT_TARGET_ESTIMATE),
    "fc/attitude":     (AttitudeState,   FMT_ATTITUDE_STATE),
    "fc/imu":          (IMUFrame,        FMT_IMU_FRAME),
    "guidance/accel":  (AccelCmd,        FMT_ACCEL_CMD),
    "control/cmd":     (ControlCmd,      FMT_CONTROL_CMD),
    "lockon/cmd":      (LockOnCmd,       FMT_LOCKON_CMD),
    "system/health":   (HealthReport,    FMT_HEALTH_REPORT),
}


@dataclass
class _TopicState:
    shm:       SharedMemory
    lock:      multiprocessing.Lock
    head:      multiprocessing.Value
    r_fd:      int
    w_fd:      int
    slot_size: int
    msg_class: type
    fmt:       str


class Bus:
    """Shared-memory message bus. Created once in the parent process before
    forking workers; all state is inherited across fork.

    Topics are pre-declared (see TOPICS). Any call with an unknown topic name
    raises KeyError immediately — that is a programming error.
    """

    def __init__(self, ring_depth: int = 8) -> None:
        self._ring_depth = ring_depth
        self._topics: dict[str, _TopicState] = {}
        for name, (msg_class, fmt) in TOPICS.items():
            slot_size = struct.calcsize(fmt)
            shm = SharedMemory(create=True, size=ring_depth * slot_size)
            lock = multiprocessing.Lock()
            head = multiprocessing.Value("i", -1)
            r_fd, w_fd = os.pipe()
            # r_fd is non-blocking so the drain in publish never blocks.
            fcntl.fcntl(r_fd, fcntl.F_SETFL, os.O_NONBLOCK)
            self._topics[name] = _TopicState(
                shm=shm, lock=lock, head=head,
                r_fd=r_fd, w_fd=w_fd,
                slot_size=slot_size, msg_class=msg_class, fmt=fmt,
            )

    def _get_state(self, topic: str) -> _TopicState:
        try:
            return self._topics[topic]
        except KeyError:
            raise KeyError(
                f"Unknown bus topic: {topic!r}. Valid topics: {sorted(TOPICS)}"
            )

    def publish(self, topic: str, msg) -> None:
        """Write msg to topic's ring and send a wakeup byte to its pipe.

        The drain+write pair is inside the lock so the pipe holds at most one
        byte regardless of publish rate (edge-triggered, not level-triggered).
        try/finally ensures the lock is always released even if os.write raises.
        """
        state = self._get_state(topic)
        state.lock.acquire()
        try:
            data = msg.pack()
            new_head = (state.head.value + 1) % self._ring_depth
            offset = new_head * state.slot_size
            state.shm.buf[offset : offset + state.slot_size] = data
            state.head.value = new_head
            try:
                os.read(state.r_fd, 1)
            except BlockingIOError:
                pass
            os.write(state.w_fd, b"\x00")
        finally:
            state.lock.release()

    def latest(self, topic: str):
        """Return the most recent message on topic, or None if never published."""
        state = self._get_state(topic)
        state.lock.acquire()
        try:
            h = state.head.value
            if h == -1:
                return None
            offset = h * state.slot_size
            data = bytes(state.shm.buf[offset : offset + state.slot_size])
        finally:
            state.lock.release()
        return state.msg_class.unpack(data)

    def close(self) -> None:
        """Unlink all shared memory and close all pipe fds.

        Must only be called by the parent process after all workers have been
        joined. os.close(r_fd) closes only the parent's fd copy; children retain
        theirs until they exit.
        """
        for state in self._topics.values():
            state.shm.close()
            state.shm.unlink()
            os.close(state.r_fd)
            os.close(state.w_fd)
