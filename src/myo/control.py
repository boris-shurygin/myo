"""Loopback command receiver; ordered episode-scoped commands and relax timeout."""
from __future__ import annotations
import json
import math
import socket
import time


class LoopbackCommandReceiver:
    def __init__(self, port: int, timeout: float = 0.5):
        if not 0 <= port <= 65535 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid command port/timeout.")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', port))
        self.socket.setblocking(False)
        self.timeout = timeout
        self.session = None
        self.sequence = -1
        self.last_received = None
        self.timed_out = False

    def poll(self, env, run_id: str, episode: int, now: float | None = None) -> bool:
        """Apply newest matching command. Sender obtains run/episode from state v2.

        Arrival time controls timeout. Physics still uses its fixed timestep;
        the timeout changes only the user trigger, retaining protective control.
        """
        now = time.monotonic() if now is None else now
        identity = (run_id, episode)
        if self.session != identity:
            self.session, self.sequence, self.last_received = identity, -1, now
            self.timed_out = False
        changed = False
        # Bound work per control frame even if a local producer floods the port.
        for _ in range(256):
            try:
                payload, _ = self.socket.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                command = json.loads(payload)
                sequence = command['sequence']
                target, trigger = command['target_angle_rad'], command['trigger']
                valid = (command['schema'] == 'myo.character_command.v1'
                         and command['run_id'] == run_id and command['episode'] == episode
                         and type(sequence) is int and sequence > self.sequence
                         and type(target) in (int, float) and math.isfinite(target)
                         and type(trigger) in (int, float) and 0 <= trigger <= 1)
                if not valid:
                    continue
            except (ValueError, KeyError, TypeError, UnicodeError):
                continue
            env.set_command(target, trigger)
            self.sequence, self.last_received, self.timed_out = sequence, now, False
            changed = True
        if self.last_received is not None and now - self.last_received >= self.timeout and not self.timed_out:
            env.set_command(env.command.requested, 0.0)
            self.timed_out, changed = True, True
        return changed

    def close(self):
        self.socket.close()
