"""
inference.py
-------------
Command-line entry point: run the full pose-estimation + action-recognition
pipeline on a video file (or webcam) and save an annotated output video.

Usage:
    # Process a video file
    python inference.py --source data/sample_videos/sample.mp4 --output outputs/result.mp4

    # Use your webcam (source = 0)
    python inference.py --source 0 --output outputs/webcam_result.mp4

    # Use a custom config
    python inference.py --source video.mp4 --config config.yaml
"""

import argparse
import time
import cv2

from src.pipeline import CrowdPoseActionPipeline


def run(args):
    pipeline = CrowdPoseActionPipeline(config_path=args.config)

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    frame_count = 0
    action_tally = {}
    start_time = time.time()

    print(f"[inference] Processing source: {args.source} -> {args.output}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, results = pipeline.process_frame(frame)

        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps_in, (w, h))

        writer.write(annotated)
        frame_count += 1

        for r in results:
            action_tally[r["action"]] = action_tally.get(r["action"], 0) + 1

        if args.show:
            cv2.imshow("Pose + Action Recognition (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  frame {frame_count} | avg FPS so far: {frame_count / elapsed:.2f}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("\n=== Summary ===")
    print(f"Frames processed : {frame_count}")
    print(f"Total time       : {elapsed:.1f}s ({frame_count / max(elapsed,1e-6):.2f} FPS)")
    print("Action label counts across all frames/people:")
    for action, count in sorted(action_tally.items(), key=lambda x: -x[1]):
        print(f"  {action:10s}: {count}")
    print(f"Output saved to  : {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/sample_videos/sample.mp4",
                         help="Video file path, or 0 for webcam")
    parser.add_argument("--output", default="outputs/result.mp4")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()
    run(args)
