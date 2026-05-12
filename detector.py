"""
Crowd Detection System using YOLOv8 + RTSP Stream
Detects and counts persons in real-time from an RTSP camera feed.
"""

import cv2
import time
import argparse
import logging
from datetime import datetime
from ultralytics import YOLO

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
PERSON_CLASS_ID = 0          # COCO class index for "person"
DEFAULT_CONF    = 0.40       # minimum confidence threshold
DEFAULT_IOU     = 0.45       # NMS IoU threshold
RECONNECT_DELAY = 5          # seconds to wait before reconnecting


# ── Core detector ─────────────────────────────────────────────────────────────
class CrowdDetector:
    """
    Wraps a YOLOv8 model and an OpenCV VideoCapture to detect
    persons from an RTSP stream.
    """

    def __init__(
        self,
        rtsp_url: str,
        model_path: str = "yolov8n.pt",
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        crowd_threshold: int = 10,
        show_window: bool = True,
        save_video: bool = False,
        output_path: str = "output.mp4",
    ):
        self.rtsp_url        = rtsp_url
        self.conf            = conf
        self.iou             = iou
        self.crowd_threshold = crowd_threshold
        self.show_window     = show_window
        self.save_video      = save_video
        self.output_path     = output_path

        log.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)

        self.cap    = None
        self.writer = None

    # ── Stream helpers ────────────────────────────────────────────────────────

    def _open_stream(self) -> bool:
        """Open (or re-open) the RTSP stream."""
        if self.cap:
            self.cap.release()

        log.info("Connecting to stream: %s", self.rtsp_url)
        # Use FFMPEG backend for better RTSP support
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low

        if not self.cap.isOpened():
            log.error("Failed to open stream.")
            return False

        log.info("Stream opened successfully.")
        return True

    def _init_writer(self, frame_w: int, frame_h: int, fps: float):
        """Initialise the video writer (called once on first frame)."""
        if self.save_video and self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, (frame_w, frame_h)
            )
            log.info("Saving output to: %s", self.output_path)

    # ── Drawing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_box(frame, x1, y1, x2, y2, conf):
        """Draw a bounding box with confidence label."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_hud(frame, count: int, fps: float, is_crowd: bool, threshold: int):
        """Draw HUD overlay: person count, FPS, crowd alert."""
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Person count
        count_color = (0, 0, 255) if is_crowd else (0, 255, 0)
        cv2.putText(
            frame, f"Persons: {count}", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, count_color, 2, cv2.LINE_AA,
        )

        # FPS
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (w - 160, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Crowd alert banner
        if is_crowd:
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 200), -1)
            msg = f"⚠  CROWD ALERT  — {count} persons detected (threshold: {threshold})"
            cv2.putText(
                frame, msg, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            frame, ts, (10, h - 60 if is_crowd else h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Start the detection loop. Press 'q' to quit."""
        if not self._open_stream():
            return

        fps_cap   = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._init_writer(frame_w, frame_h, fps_cap)

        prev_time = time.time()
        fps_display = 0.0

        log.info("Detection started. Press 'q' in the window to quit.")

        while True:
            ret, frame = self.cap.read()

            # ── Reconnect on dropped stream ───────────────────────────────────
            if not ret or frame is None:
                log.warning("Stream lost. Reconnecting in %ds…", RECONNECT_DELAY)
                time.sleep(RECONNECT_DELAY)
                if not self._open_stream():
                    continue
                continue

            # ── Inference (persons only) ──────────────────────────────────────
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                classes=[PERSON_CLASS_ID],
                verbose=False,
            )[0]

            # ── Parse detections ──────────────────────────────────────────────
            person_count = 0
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                self._draw_box(frame, x1, y1, x2, y2, conf_val)
                person_count += 1

            # ── FPS calculation ───────────────────────────────────────────────
            now       = time.time()
            fps_display = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── HUD ───────────────────────────────────────────────────────────
            is_crowd = person_count >= self.crowd_threshold
            self._draw_hud(frame, person_count, fps_display, is_crowd, self.crowd_threshold)

            # ── Log crowd events ──────────────────────────────────────────────
            if is_crowd:
                log.warning("CROWD DETECTED: %d persons", person_count)

            # ── Output ────────────────────────────────────────────────────────
            if self.writer:
                self.writer.write(frame)

            if self.show_window:
                cv2.imshow("Crowd Detection — RTSP", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit signal received.")
                    break

        self._cleanup()

    def _cleanup(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        log.info("Resources released.")


# ── CLI entry-point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time crowd detection from an RTSP stream using YOLOv8."
    )
    parser.add_argument(
        "--rtsp",
        default="rtsp://192.168.1.16:5543/live/channel0",
        help="RTSP stream URL (default: rtsp://192.168.1.16:5543/live/channel0)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model weights (default: yolov8n.pt). "
             "Options: yolov8n/s/m/l/x.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Detection confidence threshold (default: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"NMS IoU threshold (default: {DEFAULT_IOU})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Person count that triggers a crowd alert (default: 10)",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the display window (headless / server mode)",
    )
    parser.add_argument(
        "--save",
        metavar="OUTPUT.mp4",
        default=None,
        help="Save annotated video to this file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    detector = CrowdDetector(
        rtsp_url        = args.rtsp,
        model_path      = args.model,
        conf            = args.conf,
        iou             = args.iou,
        crowd_threshold = args.threshold,
        show_window     = not args.no_window,
        save_video      = args.save is not None,
        output_path     = args.save or "output.mp4",
    )
    detector.run()
