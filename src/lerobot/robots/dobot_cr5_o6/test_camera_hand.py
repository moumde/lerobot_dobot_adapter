#!/usr/bin/env python3

import logging
import time
from typing import Dict, Tuple

import cv2
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.realsense import RealSenseCameraConfig

from core.rs485.linker_hand_o6_rs485 import LinkerHandO6RS485


# ================================================================
# Configuration
# ================================================================

# ----------------
# RealSense
# ----------------

LEFT_CAMERA_SERIAL = "317222074617"
RIGHT_CAMERA_SERIAL = "254622075848"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# OpenPI output size
OUTPUT_WIDTH = 224
OUTPUT_HEIGHT = 224

# right camera -> base ROI
# [x1, y1, x2, y2]
ROI_NORM = (
    0.37,
    0.56,
    0.55,
    0.79,
)


# ----------------
# LinkerHand O6
# ----------------

HAND_ID = 39

HAND_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Serial-if00-port0"
)

HAND_BAUDRATE = 115200


# ================================================================
# Logging
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ================================================================
# OpenPI resize
# ================================================================

def resize_rgb_for_openpi(
    image: np.ndarray,
    height: int = OUTPUT_HEIGHT,
    width: int = OUTPUT_WIDTH,
) -> np.ndarray:
    """
    Match OpenPI's aspect-preserving, centered-black-padding resize.

    Input:
        H x W x 3 RGB image

    Output:
        224 x 224 x 3 RGB uint8 image

    Example:

        640 x 480
            ↓
        224 x 168
            ↓
        centered in 224 x 224 black canvas
    """

    array = np.asarray(image)

    # ------------------------------------------------------------
    # Check shape
    # ------------------------------------------------------------

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"RGB image must have HxWx3 shape, got {array.shape}"
        )

    # ------------------------------------------------------------
    # Convert dtype to uint8
    # ------------------------------------------------------------

    if array.dtype != np.uint8:

        if np.issubdtype(array.dtype, np.floating):

            maximum = (
                float(np.nanmax(array))
                if array.size
                else 0.0
            )

            if maximum <= 1.0:
                array = array * 255.0

        array = np.clip(
            array,
            0,
            255,
        ).astype(np.uint8)

    # ------------------------------------------------------------
    # Source size
    # ------------------------------------------------------------

    source_height, source_width = array.shape[:2]

    if source_height <= 0 or source_width <= 0:
        raise ValueError("RGB image is empty")

    # Already correct size
    if (
        source_height == height
        and source_width == width
    ):
        return np.ascontiguousarray(array)

    # ------------------------------------------------------------
    # Aspect preserving resize
    # ------------------------------------------------------------

    ratio = max(
        source_width / width,
        source_height / height,
    )

    resized_height = max(
        1,
        int(source_height / ratio),
    )

    resized_width = max(
        1,
        int(source_width / ratio),
    )

    resized = cv2.resize(
        array,
        (
            resized_width,
            resized_height,
        ),
        interpolation=(
            cv2.INTER_AREA
            if ratio >= 1.0
            else cv2.INTER_LINEAR
        ),
    )

    # ------------------------------------------------------------
    # Black canvas
    # ------------------------------------------------------------

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    # ------------------------------------------------------------
    # Center
    # ------------------------------------------------------------

    top = (
        height - resized_height
    ) // 2

    left = (
        width - resized_width
    ) // 2

    canvas[
        top : top + resized_height,
        left : left + resized_width,
    ] = resized

    return np.ascontiguousarray(canvas)


# ================================================================
# ROI
# ================================================================

def crop_roi_norm(
    image: np.ndarray,
    roi_norm: Tuple[
        float,
        float,
        float,
        float,
    ],
) -> np.ndarray:
    """
    Crop normalized ROI.

    roi_norm:
        [x1, y1, x2, y2]

    Coordinates are normalized to [0, 1].
    """

    if image.ndim != 3:
        raise ValueError(
            f"Expected HxWx3 image, got {image.shape}"
        )

    height, width = image.shape[:2]

    x1_norm, y1_norm, x2_norm, y2_norm = roi_norm

    # ------------------------------------------------------------
    # Validate ROI
    # ------------------------------------------------------------

    if not (
        0.0 <= x1_norm < x2_norm <= 1.0
        and
        0.0 <= y1_norm < y2_norm <= 1.0
    ):
        raise ValueError(
            f"Invalid roi_norm: {roi_norm}"
        )

    # ------------------------------------------------------------
    # Convert normalized -> pixel
    # ------------------------------------------------------------

    x1 = int(x1_norm * width)
    y1 = int(y1_norm * height)

    x2 = int(x2_norm * width)
    y2 = int(y2_norm * height)

    # Make sure we remain inside image
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))

    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    logger.debug(
        "ROI pixel: "
        f"({x1}, {y1}) -> ({x2}, {y2})"
    )

    return np.ascontiguousarray(
        image[y1:y2, x1:x2]
    )


# ================================================================
# Camera initialization
# ================================================================

def create_cameras():

    cameras_config = {

        "left_wrist_0_rgb": RealSenseCameraConfig(
            serial_number_or_name=LEFT_CAMERA_SERIAL,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),

        "right_wrist_0_rgb": RealSenseCameraConfig(
            serial_number_or_name=RIGHT_CAMERA_SERIAL,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
    }

    return make_cameras_from_configs(
        cameras_config
    )


# ================================================================
# Main
# ================================================================

def main():

    cameras = {}
    hand = None

    try:

        # ========================================================
        # 1. Create cameras
        # ========================================================

        print()
        print("=" * 70)
        print("STEP 1: Create RealSense cameras")
        print("=" * 70)

        cameras = create_cameras()

        print(
            f"Detected configured cameras: "
            f"{len(cameras)}"
        )

        for name in cameras:
            print(
                f"  - {name}"
            )

        # ========================================================
        # 2. Connect cameras
        # ========================================================

        print()
        print("=" * 70)
        print("STEP 2: Connect cameras")
        print("=" * 70)

        for name, camera in cameras.items():

            print(
                f"Connecting {name}..."
            )

            camera.connect()

            print(
                f"  connected = "
                f"{camera.is_connected}"
            )

        # ========================================================
        # 3. Connect O6
        # ========================================================

        print()
        print("=" * 70)
        print("STEP 3: Connect LinkerHand O6")
        print("=" * 70)

        print(
            f"hand_id  = {HAND_ID}"
        )

        print(
            f"port     = {HAND_PORT}"
        )

        print(
            f"baudrate = {HAND_BAUDRATE}"
        )

        hand = LinkerHandO6RS485(
            hand_id=HAND_ID,
            modbus_port=HAND_PORT,
            baudrate=HAND_BAUDRATE,
        )

        print(
            "O6 connection successful."
        )

        # ========================================================
        # 4. Read initial O6 state
        # ========================================================

        print()
        print("=" * 70)
        print("STEP 4: Read initial O6 state")
        print("=" * 70)

        state = hand.get_state()

        print(
            f"O6 state = {state}"
        )

        print(
            f"State length = {len(state)}"
        )

        if len(state) != 6:
            raise RuntimeError(
                "O6 state length is not 6!"
            )

        print(
            "O6 state read: OK"
        )

        # ========================================================
        # 5. Read O6 information
        # ========================================================

        print()
        print("=" * 70)
        print("STEP 5: Read O6 device information")
        print("=" * 70)

        try:

            print(
                "Device number:",
                hand.get_device_number(),
            )

            print(
                "Hardware version:",
                hand.get_hardware_version(),
            )

            print(
                "Software version:",
                hand.get_software_version(),
            )

            print(
                "Mechanical version:",
                hand.get_mechanical_version(),
            )

            print(
                "Hand direction:",
                hand.get_hand_direction(),
            )

        except Exception as exc:

            logger.warning(
                f"Failed to read optional O6 information: "
                f"{exc}"
            )

        # ========================================================
        # 6. Start test loop
        # ========================================================

        print()
        print("=" * 70)
        print("STEP 6: Start Camera + O6 test")
        print("=" * 70)

        print()
        print("Windows:")
        print(
            "  left_wrist_0_rgb  = left camera"
        )
        print(
            "  right_wrist_0_rgb = right camera"
        )
        print(
            "  base_0_rgb        = right camera ROI"
        )
        print()
        print(
            "Press 'q' to quit."
        )
        print()

        frame_count = 0

        start_time = time.perf_counter()

        while True:

            # ====================================================
            # Camera
            # ====================================================

            left_raw = cameras[
                "left_wrist_0_rgb"
            ].read_latest()

            right_raw = cameras[
                "right_wrist_0_rgb"
            ].read_latest()

            # ----------------------------------------------------
            # Check raw images
            # ----------------------------------------------------

            if left_raw is None:
                raise RuntimeError(
                    "Left camera returned None"
                )

            if right_raw is None:
                raise RuntimeError(
                    "Right camera returned None"
                )

            # ----------------------------------------------------
            # Print raw image shape once
            # ----------------------------------------------------

            if frame_count == 0:

                print()
                print(
                    "Raw camera images:"
                )

                print(
                    "  left :",
                    left_raw.shape,
                    left_raw.dtype,
                )

                print(
                    "  right:",
                    right_raw.shape,
                    right_raw.dtype,
                )

            # ====================================================
            # OpenPI resize
            # ====================================================

            left = resize_rgb_for_openpi(
                left_raw,
                height=OUTPUT_HEIGHT,
                width=OUTPUT_WIDTH,
            )

            right = resize_rgb_for_openpi(
                right_raw,
                height=OUTPUT_HEIGHT,
                width=OUTPUT_WIDTH,
            )

            # ====================================================
            # Right -> Base ROI
            # ====================================================

            base_roi = crop_roi_norm(
                right,
                ROI_NORM,
            )

            base = resize_rgb_for_openpi(
                base_roi,
                height=OUTPUT_HEIGHT,
                width=OUTPUT_WIDTH,
            )

            if frame_count == 0:

                print()
                print(
                    "Output camera images:"
                )

                print(
                    "  left :",
                    left.shape,
                    left.dtype,
                )

                print(
                    "  right:",
                    right.shape,
                    right.dtype,
                )

                print(
                    "  base :",
                    base.shape,
                    base.dtype,
                )

            # ====================================================
            # O6 state
            # ====================================================

            o6_state = hand.get_state()

            if len(o6_state) != 6:
                raise RuntimeError(
                    f"Invalid O6 state: {o6_state}"
                )

            # ====================================================
            # Display
            # ====================================================

            left_bgr = cv2.cvtColor(
                left,
                cv2.COLOR_RGB2BGR,
            )

            right_bgr = cv2.cvtColor(
                right,
                cv2.COLOR_RGB2BGR,
            )

            base_bgr = cv2.cvtColor(
                base,
                cv2.COLOR_RGB2BGR,
            )

            # ----------------------------------------------------
            # Add labels
            # ----------------------------------------------------

            cv2.putText(
                left_bgr,
                "LEFT",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                right_bgr,
                "RIGHT",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                base_bgr,
                "BASE ROI",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # ----------------------------------------------------
            # Show
            # ----------------------------------------------------

            cv2.imshow(
                "left_wrist_0_rgb",
                left_bgr,
            )

            cv2.imshow(
                "right_wrist_0_rgb",
                right_bgr,
            )

            cv2.imshow(
                "base_0_rgb",
                base_bgr,
            )

            # ====================================================
            # Print O6 state
            # ====================================================

            elapsed = (
                time.perf_counter()
                - start_time
            )

            frame_count += 1

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"\r"
                f"Frame={frame_count:06d} | "
                f"FPS={fps:5.1f} | "
                f"O6=["
                f"{o6_state[0]:3d}, "
                f"{o6_state[1]:3d}, "
                f"{o6_state[2]:3d}, "
                f"{o6_state[3]:3d}, "
                f"{o6_state[4]:3d}, "
                f"{o6_state[5]:3d}"
                f"]",
                end="",
                flush=True,
            )

            # ====================================================
            # Keyboard
            # ====================================================

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print()
        print(
            "\nKeyboard interrupt."
        )

    except Exception as exc:

        print()
        print()
        print("=" * 70)
        print("TEST FAILED")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    finally:

        # ========================================================
        # Disconnect cameras
        # ========================================================

        print()
        print()
        print("=" * 70)
        print("Cleanup")
        print("=" * 70)

        for name, camera in cameras.items():

            try:

                if camera.is_connected:
                    camera.disconnect()

                print(
                    f"{name}: disconnected"
                )

            except Exception as exc:

                print(
                    f"{name}: disconnect failed: "
                    f"{exc}"
                )

        # ========================================================
        # Disconnect O6
        # ========================================================

        if hand is not None:

            try:

                hand.close()

                print(
                    "O6: disconnected"
                )

            except Exception as exc:

                print(
                    f"O6: disconnect failed: "
                    f"{exc}"
                )

        cv2.destroyAllWindows()

        print()
        print(
            "Test finished."
        )


# ================================================================
# Entry
# ================================================================

if __name__ == "__main__":
    main()