#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run the video generation services')
    parser.add_argument('--web-only', action='store_true', help='Run only the web service')
    parser.add_argument('--rmq-only', action='store_true', help='Run only the RMQ consumer')
    parser.add_argument('--port', type=int, default=5000, help='Port for the web service')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/Wav2Lip.pth', 
                        help='Path to the Wav2Lip model checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables
    os.environ["PORT"] = str(args.port)
    os.environ["WEBSERVICE_URL"] = f"http://localhost:{args.port}"
    os.environ["CHECKPOINT_PATH"] = args.checkpoint_path
    
    # Create directories
    os.makedirs("./temp", exist_ok=True)
    os.makedirs("./out_video_folder", exist_ok=True)
    
    # Start the web service
    if not args.rmq_only:
        print("Starting web service...")
        web_process = subprocess.Popen([sys.executable, "webservice.py"])
        time.sleep(2)  # Give the web service time to start
    else:
        web_process = None
    
    # Start the RMQ consumer
    if not args.web_only:
        print("Starting RMQ consumer...")
        try:
            rmq_process = subprocess.Popen([sys.executable, "inference.py"])
            rmq_process.wait()
        except KeyboardInterrupt:
            print("Stopping services...")
        finally:
            if web_process:
                web_process.terminate()
    else:
        # If only running web service, wait for it
        if web_process:
            try:
                web_process.wait()
            except KeyboardInterrupt:
                print("Stopping web service...")
                web_process.terminate()

if __name__ == "__main__":
    main()
