import time

import numpy as np

from lerobot.cameras import ColorMode, Cv2Rotation
from lerobot.cameras.realsense import (
    RealSenseCamera,
    RealSenseCameraConfig,
)


CAMERA_CONFIG = RealSenseCameraConfig(
    serial_number_or_name="262422074985",
    color_mode=ColorMode.RGB,
    use_rgb=True,
    use_depth=False,
    rotation=Cv2Rotation.NO_ROTATION,
    warmup_s=1,
    fps=30,
    width=640,
    height=480,
)


def main():
    print("=" * 60)
    print("D435 Camera Configuration")
    print("=" * 60)

    print(CAMERA_CONFIG)

    camera = RealSenseCamera(CAMERA_CONFIG)

    print("\nConnecting...")
    camera.connect()

    print("Connected:", camera.is_connected)

    try:
        print("\nCamera runtime information:")
        print("  FPS   :", camera.fps)
        print("  Width :", camera.width)
        print("  Height:", camera.height)

        print("\nTesting RGB frame...")

        image = camera.read()

        print("  type :", type(image))
        print("  shape:", image.shape)
        print("  dtype:", image.dtype)
        print("  min  :", image.min())
        print("  max  :", image.max())

        assert isinstance(image, np.ndarray)
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8

        print("\n✓ RGB frame validation passed")

        # --------------------------------------------------
        # Test continuous capture
        # --------------------------------------------------

        print("\nTesting continuous capture...")

        num_frames = 100
        timestamps = []

        for i in range(num_frames):
            t0 = time.perf_counter()

            frame = camera.read()

            t1 = time.perf_counter()

            assert frame.shape == (480, 640, 3)
            assert frame.dtype == np.uint8

            timestamps.append(t1)

        timestamps = np.asarray(timestamps)

        intervals = np.diff(timestamps)

        duration = timestamps[-1] - timestamps[0]
        fps = 1.0 / intervals.mean()

        print("\nContinuous capture result:")
        print("  Frames       :", num_frames)
        print("  Duration     :", f"{duration:.3f} s")
        print("  Average FPS  :", f"{fps:.3f}")
        print("  Mean interval:", f"{intervals.mean() * 1000:.3f} ms")
        print("  Std interval :", f"{intervals.std() * 1000:.3f} ms")
        print("  Min interval :", f"{intervals.min() * 1000:.3f} ms")
        print("  Max interval :", f"{intervals.max() * 1000:.3f} ms")

        print("\n✓ Continuous capture validation passed")

    finally:
        print("\nDisconnecting...")
        camera.disconnect()

    print("\n" + "=" * 60)
    print("✓ D435 CAMERA CONFIGURATION VALID")
    print("=" * 60)


if __name__ == "__main__":
    main()