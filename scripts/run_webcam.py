#!/usr/bin/env python3
"""
Real-Time Eyeglass Detection - Webcam Script

This script runs the eyeglass detector on a live webcam feed.
Use --help for available options.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from glass_detection import EyeglassDetector


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time eyeglass detection using webcam.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.15,
        help="Detection threshold (0.0-1.0). Higher = stricter detection"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Show debug windows with detection regions"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    
    print("=" * 50)
    print("Real-Time Eyeglass Detection")
    print("=" * 50)
    print(f"Camera ID: {args.camera}")
    print(f"Threshold: {args.threshold}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 50)
    print()
    
    # Initialize detector
    detector = EyeglassDetector(threshold=args.threshold)
    
    # Run webcam detection
    try:
        detector.run_webcam(
            camera_id=args.camera,
            show_debug=args.debug
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
