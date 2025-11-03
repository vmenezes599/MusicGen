"""Client script to send text-to-music requests to a MusicGen server."""

import argparse
from pathlib import Path
from zipfile import ZipFile
import requests


class Client:
    """Client class for sending text-to-music requests to a MusicGen server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8190):
        """
        Initialize the MusicGen client.

        Args:
            host: Server hostname or IP address
            port: Server port number
        """
        self._host = host
        self._port = port
        self._base_url = f"http://{host}:{port}"
        self._default_timeout = 600  # 10 minutes for music generation

    def generate_music(
        self,
        prompt: str,
        seed: int,
        seconds: float = 30.0,
        output_path: str | Path | None = None,
    ) -> bytes | bool | None:
        """
        Generate music from text prompt.

        Args:
            prompt: Text prompt for music generation
            seed: Random seed for generation
            seconds: Duration in seconds (default: 30.0)
            output_path: Optional output path to save the ZIP file containing MP3s

        Returns:
            If output_path is provided, returns True on success, False on failure.
            If output_path is None, returns the ZIP file bytes on success, None on failure.
        """
        # Create output directory if output_path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"{self._base_url}/ttm"

        # Prepare form data
        data = {
            "prompt": prompt,
            "seed": seed,
            "seconds": seconds,
        }

        try:
            response = requests.post(url, data=data, stream=True, timeout=self._default_timeout)

            if response.status_code == 200:
                if output_path:
                    # Save ZIP to file
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Music ZIP saved to {output_path}")
                    
                    # Unzip the file to the output directory
                    output_dir = output_path.parent / output_path.stem
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    with ZipFile(output_path, "r") as zipf:
                        zipf.extractall(output_dir)
                    
                    print(f"Music files extracted to {output_dir}")
                    
                    # Delete the ZIP file after extraction
                    output_path.unlink()
                    print(f"Deleted ZIP file: {output_path}")
                    
                    return True
                else:
                    # Return ZIP bytes
                    return response.content
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return False if output_path else None

        except (requests.RequestException, IOError) as e:
            print(f"Error during request: {e}")
            return False if output_path else None

    def test_connection(self) -> bool:
        """
        Test connection to the MusicGen server.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self._base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


def main():
    """Main function to send a request to the MusicGen server via command line."""
    parser = argparse.ArgumentParser(description="MusicGen Client - Send text-to-music generation requests")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8189, help="Server port")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for music generation")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for generation")
    parser.add_argument("--seconds", type=float, default=30.0, help="Duration in seconds (default: 30.0)")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for generated music ZIP file",
    )
    args = parser.parse_args()

    # Use the Client class for CLI functionality
    client = Client(host=args.host, port=args.port)
    success = client.generate_music(
        prompt=args.prompt,
        seed=args.seed,
        seconds=args.seconds,
        output_path=args.output_path,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    # Example usage:
    # python client.py --host 127.0.0.1 --port 8189 --prompt "upbeat electronic music" --seed 42 --seconds 30.0 --output_path ./output/music.zip
    main()
