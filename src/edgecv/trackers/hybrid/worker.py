"""Detector worker entrypoint for MAFiD hybrid tracker (MAFiD spec §5.7).

Runs in a spawned child process. Constructs the NN detector via factory, builds
CF filters via a matching CF instance, and publishes candidates to the payload
channel. Never receives live tracker state — only config and SHM names.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from edgecv.core.bbox import BoundingBox
from edgecv.runtime.shm.frame_ring import FrameRing
from edgecv.runtime.shm.payload import PayloadChannel
from edgecv.runtime.shm.search_roi import SearchROIChannel
from edgecv.runtime.worker import request_death_with_parent
from edgecv.trackers.hybrid.serialise import _serialise_filter_state

log = logging.getLogger("edgecv.hybrid.worker")


def _detector_main(
    detector_factory,
    detector_config: dict,
    cf_tracker_cls,
    cf_kwargs: dict,
    nn_calibrator,
    nn_confidence_floor: float,
    fr_name: str,
    roi_name: str,
    payload_name: str,
    max_h: int,
    max_w: int,
    max_c: int,
    frame_slots: int,
    payload_capacity: int,
    stop_event,
    template_gen=None,
) -> None:
    """Runs in the spawned child process.

    Everything that touches a backend is constructed HERE, inside the child.
    The parent never even imports a backend model.
    """
    request_death_with_parent()

    # --- Construct components INSIDE the child ---
    detector = detector_factory(detector_config)
    cf_builder = cf_tracker_cls(**cf_kwargs)

    # --- Attach SHM (never create/unlink) ---
    frame_ring = FrameRing.attach(
        fr_name, slots=frame_slots,
        max_h=max_h, max_w=max_w, max_c=max_c,
    )
    roi_channel = SearchROIChannel.attach(roi_name)
    payload = PayloadChannel.attach(
        payload_name, capacity_bytes=payload_capacity,
    )

    last_processed_seq = 0  # skip frames we've already seen
    last_template_gen = template_gen.value if template_gen is not None else 0

    try:
        while not stop_event.is_set():
            # 0. Check for template refresh signal (parent→worker mutual assistance)
            if template_gen is not None:
                current_gen = template_gen.value
                if current_gen != last_template_gen:
                    last_template_gen = current_gen
                    if hasattr(detector, 'request_refresh'):
                        detector.request_refresh()

            # 1. Read latest frame
            fr = frame_ring.read_latest()
            if fr is None:
                time.sleep(0.001)
                continue
            frame, fr_seq, fr_ts = fr

            # Skip if we already processed this frame (no new frame yet)
            if fr_seq <= last_processed_seq:
                time.sleep(0.001)
                continue
            last_processed_seq = fr_seq

            # 2. Read latest search ROI
            roi = roi_channel.read_latest()
            if roi is None:
                continue  # caller hasn't published yet

            detect_time = time.monotonic()

            # 3. Crop + detect (the NN part)
            det_out = detector.detect(frame, roi)

            if len(det_out.boxes) == 0:
                continue  # no detection → loop

            # 4. NN confidence gate (pre-filter before expensive filter build)
            best_idx = int(np.argmax(det_out.scores))
            best_nn_score = float(det_out.scores[best_idx])
            nn_conf = nn_calibrator.calibrate(best_nn_score)
            if nn_conf < nn_confidence_floor:
                continue  # detection not confident enough

            # 5. Build CF filter from the best detection
            best_box = BoundingBox(
                x=float(det_out.boxes[best_idx, 0]),
                y=float(det_out.boxes[best_idx, 1]),
                w=float(det_out.boxes[best_idx, 2]),
                h=float(det_out.boxes[best_idx, 3]),
            )

            # Skip degenerate boxes (near-zero or negative dimensions)
            if best_box.w <= 0.0 or best_box.h <= 0.0:
                continue
            min_dim = 4.0 / min(max_h, max_w)  # at least 4 pixels
            if best_box.w < min_dim or best_box.h < min_dim:
                continue

            filter_state = cf_builder.build_filter(frame, best_box)

            # 6. Publish candidate to payload channel
            payload_arrays = _serialise_filter_state(filter_state)
            payload_arrays["detector_out_boxes"] = det_out.boxes
            payload_arrays["detector_out_scores"] = det_out.scores
            payload_arrays["detect_time"] = np.array([detect_time], np.float64)

            payload.publish(payload_arrays, fr_seq)

    finally:
        # Clean shutdown — release backend resources before exit
        detector.close()
        cf_builder.close()
        frame_ring.close(unlink=False)
        roi_channel.close(unlink=False)
        payload.close(unlink=False)
