"""Test script for the MusicGen Client class."""

from client import Client


def main():
    """Main function to test the MusicGen client with multiple requests."""
    # Initialize the client
    client = Client(host="127.0.0.1", port=8190)

    # Test connection first
    if not client.test_connection():
        print("Warning: Could not connect to server. Proceeding with requests anyway...")

    # List of music generation prompts
    prompt_list = [
        "upbeat electronic dance music with energetic synths",
        "calm ambient meditation music with soft pads",
        "epic orchestral cinematic soundtrack",
        "funky jazz fusion with saxophone and bass",
        "dark atmospheric horror background music",
        "cheerful acoustic guitar folk melody",
        "intense rock music with electric guitars",
        "relaxing lo-fi hip hop beats",
        "dramatic suspenseful thriller soundtrack",
        "tropical island reggae with steel drums",
    ]

    # Simple test with just one prompt
    simple_prompt = "peaceful piano melody with gentle strings"

    prompt_list = [simple_prompt]

    # Create test requests
    test_requests = []
    for i, prompt in enumerate(prompt_list):
        test_request = {
            "prompt": prompt,
            "seed": 42 + i,  # Different seed for each request
            "seconds": 30.0,
            "output_path": "output",
        }
        test_requests.append(test_request)

    # Process each request
    for i, request in enumerate(test_requests):
        print(f"Processing request {i+1}/{len(test_requests)}")
        print(f"Prompt: {request['prompt']}")

        # Make the actual request
        success = client.generate_music(
            prompt=request["prompt"],
            seed=request["seed"],
            seconds=request["seconds"],
            output_path=request["output_path"],
        )

        if success:
            print(f"✓ Successfully generated music for request {i+1}")
        else:
            print(f"✗ Failed to generate music for request {i+1}")


if __name__ == "__main__":
    main()
