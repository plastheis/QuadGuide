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
    ArmCmd,           FMT_ARM_CMD,
)

__all__ = ["Bus", "TOPICS"]

TOPICS: dict[str, tuple[type, str]] = {
    "ccv_tracker/estimate": (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "ncv_tracker/estimate": (TrackerEstimate, FMT_TRACKER_ESTIMATE),
    "target/estimate":      (TargetEstimate,  FMT_TARGET_ESTIMATE),
    "fc/attitude":          (AttitudeState,   FMT_ATTITUDE_STATE),
    "fc/imu":               (IMUFrame,        FMT_IMU_FRAME),
    "guidance/accel":       (AccelCmd,        FMT_ACCEL_CMD),
    "control/cmd":          (ControlCmd,      FMT_CONTROL_CMD),
    "lockon/cmd":           (LockOnCmd,       FMT_LOCKON_CMD),
    "system/health":        (HealthReport,    FMT_HEALTH_REPORT),
    "arm/cmd":              (ArmCmd,          FMT_ARM_CMD),
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
        data = msg.pack()  # pure computation — outside the lock to minimise hold time
        state.lock.acquire()
        try:
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
            try:
                state.shm.unlink()
            except FileNotFoundError:
                pass
            for fd in (state.r_fd, state.w_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def subscribe_one(self, topic: str):
        """Block until a new message is published on topic, then return it.

        IMPORTANT: the lock is NOT held during the blocking select.select call.
        Holding it would deadlock — publish cannot acquire the lock to complete
        its drain+write. The lock is only acquired briefly inside latest().

        Constraint: at most one process may block on a given topic at a time.
        The pipe holds one wakeup byte; two concurrent subscribers on the same
        topic means one will miss a wakeup signal.

        The retry loop handles a race where publish() drains the wakeup byte
        (under lock) between select returning and our os.read — EAGAIN means
        a concurrent publish beat us to it; re-enter select and try again.
        """
        state = self._get_state(topic)
        while True:
            select.select([state.r_fd], [], [])
            try:
                os.read(state.r_fd, 1)
                break
            except BlockingIOError:
                continue  # concurrent publish drained the byte; re-select
        return self.latest(topic)

    def subscribe_any(self, topics: list[str]) -> tuple[str, object]:
        """Block until any of the listed topics receives a publish.

        Returns (topic_name, msg). Only the first ready fd is consumed per call.
        If multiple topics fire simultaneously, select.select may return multiple
        ready fds, but only ready[0] is processed. The remaining fds retain their
        wakeup bytes and will fire immediately on the next call — this is correct,
        not a missed message.

        The retry loop handles a race where publish() drains the wakeup byte
        (under lock) between select returning and our os.read — EAGAIN means
        a concurrent publish beat us to it; re-enter select and try again.
        """
        states = [self._get_state(t) for t in topics]  # KeyError if any unknown
        r_fds = [s.r_fd for s in states]
        while True:
            ready, _, _ = select.select(r_fds, [], [])
            idx = r_fds.index(ready[0])
            try:
                os.read(states[idx].r_fd, 1)
                break
            except BlockingIOError:
                continue  # concurrent publish drained the byte; re-select
        return topics[idx], self.latest(topics[idx])

    def detach(self) -> None:
        """Close this process's fd references to shared memory and pipes.

        Called by worker processes in their SIGTERM handler — NOT by the parent.
        Uses shm.close() only, never shm.unlink(). Only the parent (Bus.close())
        owns the unlink lifecycle.
        """
        for state in self._topics.values():
            state.shm.close()
            for fd in (state.r_fd, state.w_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass
