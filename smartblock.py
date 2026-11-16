import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

from pathlib import Path
from datetime import datetime
from getpass import getpass
import queue
import sys

import cv2
import numpy as np

PROJECT_ROOT = next(
    (p for p in (Path.cwd(), *Path.cwd().parents)
     if (p / "python-dvr" / "dvrip.py").is_file()),
    Path.cwd(),
)

CAMERA_IP = "150.165.52.38"
CAMERA_PORT = 34567
CAMERA_USER = "admin"
RTSP_URL = "rtsp://127.0.0.1:8554/lab_cam"
CAPTURES_DIR = PROJECT_ROOT / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

ROI_V2 = (576, 21, 1344, 1058)

def classify_intensity(base_color: str, s: float, v: float) -> tuple[str, tuple[int, int, int]]:
    color_bgr_map = {
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
    }

    if v < 105:
        intensity = "Dark"
    elif v > 180 and s < 145:
        intensity = "Light"
    else:
        intensity = "Medium"

    return f"{intensity} {base_color}", color_bgr_map[base_color]


def detect_and_draw_colors(frame_bgr: np.ndarray, min_area: int = 900) -> np.ndarray:
    annotated_frame = frame_bgr.copy()
    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Higher minimum S (90) and V (65) ignore skin tones and dark shadows in the background
    color_ranges = {
        "Red": [
            ((0, 95, 65), (12, 255, 255)),
            ((165, 95, 65), (179, 255, 255)),
        ],
        "Green": [
            ((35, 90, 65), (85, 255, 255)),
        ],
        "Blue": [
            ((90, 95, 65), (135, 255, 255)),
        ],
    }

    open_kernel = np.ones((5, 5), np.uint8)
    shrink_kernel = np.ones((7, 7), np.uint8)

    for base_color, bounds_list in color_ranges.items():
        color_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for lower, upper in bounds_list:
            color_mask = cv2.bitwise_or(color_mask, cv2.inRange(hsv_frame, lower, upper))

        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel)
        color_mask = cv2.erode(color_mask, shrink_kernel, iterations=1)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            obj_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, thickness=cv2.FILLED)
            obj_mask = cv2.bitwise_and(obj_mask, color_mask)

            mean_h, mean_s, mean_v, _ = cv2.mean(hsv_frame, mask=obj_mask)
            label, box_color = classify_intensity(base_color, mean_s, mean_v)

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 2)
            text = f"{label} (H:{int(mean_h)})"
            cv2.putText(
                annotated_frame,
                text,
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA,
            )

    return annotated_frame


def prepare_image(image_bgr: np.ndarray, roi: tuple[int, int, int, int] = ROI_V2, size: int = 256):
    x1, y1, x2, y2 = roi
    height, width = image_bgr.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("ROI is out of frame bounds.")

    crop = image_bgr[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    scale_factor = min(size / crop_w, size / crop_h)
    new_width = max(1, round(crop_w * scale_factor))
    new_height = max(1, round(crop_h * scale_factor))
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_AREA)

    standardized = np.full((size, size, 3), 128, dtype=np.uint8)
    left_offset = (size - new_width) // 2
    top_offset = (size - new_height) // 2
    standardized[top_offset:top_offset + new_height, left_offset:left_offset + new_width] = resized

    rgb = cv2.cvtColor(standardized, cv2.COLOR_BGR2RGB)
    model_input = rgb.astype(np.float32) / 255.0
    return standardized, model_input

def create_display_window(window_name: str, width: int = 960, height: int = 540) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.waitKey(1)


def run_cctv_stream(password: str) -> list[np.ndarray]:
    import av
    dvrip_dir = PROJECT_ROOT / "python-dvr"
    if not (dvrip_dir / "dvrip.py").is_file():
        raise FileNotFoundError("Library python-dvr/dvrip.py not found in the project.")
    sys.path.insert(0, str(dvrip_dir))
    from dvrip import DVRIPCam

    camera = DVRIPCam(
        CAMERA_IP,
        user=CAMERA_USER,
        password=password,
        port=CAMERA_PORT,
    )
    packet_queue: queue.Queue[bytes] = queue.Queue(maxsize=120)
    stream_started = {"active": False}
    recorded_frames: list[np.ndarray] = []

    if not camera.login():
        camera.close()
        raise ConnectionError("Camera refused authentication.")

    def on_packet(data: bytes, meta: dict, _user) -> None:
        if not data or meta.get("frame") not in ("I", "P"):
            return
        if not stream_started["active"]:
            if meta.get("frame") != "I":
                return
            stream_started["active"] = True
        try:
            packet_queue.put_nowait(data)
        except queue.Full:
            pass

    codec_context = av.CodecContext.create("hevc", "r")
    camera.start_monitor(on_packet, stream="Main")

    window_name = "SmartBlock Live Feed (CCTV)"
    create_display_window(window_name)
    print("Streaming CCTV feed. Press 'q' on the video window to stop.")

    try:
        while True:
            try:
                raw_data = packet_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            packet = av.Packet(raw_data)
            for av_frame in codec_context.decode(packet):
                bgr_frame = av_frame.to_ndarray(format="bgr24")
                annotated = detect_and_draw_colors(bgr_frame)
                recorded_frames.append(annotated)

                cv2.imshow(window_name, annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.stop_monitor()
        camera.close()
        cv2.destroyAllWindows()

    return recorded_frames


def run_opencv_stream(source: int | str, window_name: str) -> list[np.ndarray]:
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    recorded_frames: list[np.ndarray] = []
    create_display_window(window_name)
    print(f"Streaming feed ({source}). Press 'q' on the video window to stop.")

    try:
        while True:
            success, bgr_frame = capture.read()
            if not success:
                print("Failed to grab frame from stream.")
                break

            annotated = detect_and_draw_colors(bgr_frame)
            recorded_frames.append(annotated)

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return recorded_frames

def save_recording(frames: list[np.ndarray], fps: int = 20) -> None:
    import av

    if not frames:
        print("No frames captured to save.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = CAPTURES_DIR / f"smartblock_feed_{timestamp}.mp4"
    image_path = CAPTURES_DIR / f"smartblock_last_frame_{timestamp}.png"

    height, width = frames[0].shape[:2]

    even_width = width - (width % 2)
    even_height = height - (height % 2)

    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = even_width
    stream.height = even_height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "fast"}

    for frame in frames:
        cropped = frame[:even_height, :even_width]
        av_frame = av.VideoFrame.from_ndarray(cropped, format="bgr24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    cv2.imwrite(str(image_path), frames[-1])
    print(f"Video saved to: {video_path}")
    print(f"Final frame saved to: {image_path}")

def main() -> None:
    print("Project Root:", PROJECT_ROOT)
    print("1 - CCTV camera (DVR-IP)")
    print("2 - Generic USB camera")
    print(f"3 - CCTV via RTSP ({RTSP_URL})")
    choice = input("Select camera source (1, 2, or 3): ").strip()

    if choice == "1":
        print("Target CCTV Camera:", CAMERA_IP)
        camera_password = getpass("IP Camera password: ")
        try:
            frames = run_cctv_stream(camera_password)
        finally:
            del camera_password
    elif choice == "2":
        frames = run_opencv_stream(0, "SmartBlock Live Feed (USB)")
    elif choice == "3":
        frames = run_opencv_stream(RTSP_URL, "SmartBlock Live Feed (RTSP)")
    else:
        raise ValueError("Invalid option. Please enter 1, 2, or 3.")

    if not frames:
        raise RuntimeError("No frames were captured during the session.")

    save_choice = input("Do you want to save the recorded result? (y/n): ").strip().lower()
    if save_choice in ("y", "yes"):
        save_recording(frames)

    last_frame = frames[-1]
    if last_frame.shape[1] >= ROI_V2[2] and last_frame.shape[0] >= ROI_V2[3]:
        image_256, autoencoder_input = prepare_image(last_frame)
        print("Prepared image shape:", image_256.shape)
        print("Autoencoder input shape:", autoencoder_input.shape)
        print("Autoencoder input dtype:", autoencoder_input.dtype)


if __name__ == "__main__":
    main()
