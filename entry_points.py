import logging
import subprocess
import tempfile
import os
import sys
import shutil
from pathlib import Path
from zipfile import ZipFile

import asyncio
from fastapi import Form, HTTPException, BackgroundTasks, APIRouter
from fastapi.responses import FileResponse

router = APIRouter()


async def process_music_with_subprocess(
    prompt: str,
    seed: int,
    seconds: float,
    background_tasks: BackgroundTasks,
):
    """Process music generation using subprocess for automatic resource cleanup"""

    # Create temporary directory for output files
    temp_dir = None
    temp_zip = None

    try:
        # Create temp directory for music files
        temp_dir = tempfile.mkdtemp(prefix="musicgen_")
        logging.info("Created temp directory: %s", temp_dir)

        # Run MusicGen processor as subprocess
        musicgen_processor_path = os.path.join(os.path.dirname(__file__), "musicgen_processor.py")
        cmd = [
            sys.executable,
            musicgen_processor_path,
            "--prompt",
            prompt,
            "--seed",
            str(seed),
            "--output_dir",
            temp_dir,
            "--seconds",
            str(seconds),
        ]

        logging.info("Running MusicGen subprocess: %s", " ".join(cmd))

        # Run subprocess asynchronously to avoid blocking the event loop
        # This allows health checks and other requests to be processed during music generation
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=600)  # 10 minute timeout
        except asyncio.TimeoutError as e:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(cmd, 600) from e

        if process.returncode != 0:
            stderr_text = stderr.decode("utf-8") if stderr else ""
            logging.error("MusicGen subprocess failed: %s", stderr_text)
            raise HTTPException(status_code=500, detail="Music generation failed")

        # Log any stderr output even on success (warnings, etc.)
        if stderr:
            stderr_text = stderr.decode("utf-8")
            logging.info("MusicGen subprocess stderr: %s", stderr_text)

        # Check if any files were created
        output_files = list(Path(temp_dir).glob("*.mp3"))
        if not output_files:
            raise HTTPException(status_code=500, detail="No music files were generated")

        logging.info("Generated %d music files", len(output_files))

        # Create ZIP file with all generated music
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="music_")
        temp_zip.close()

        with ZipFile(temp_zip.name, "w") as zipf:
            for music_file in output_files:
                zipf.write(music_file, music_file.name)
                logging.info("Added to zip: %s", music_file.name)

        logging.info("Created ZIP file: %s", temp_zip.name)

        # Add cleanup task to background
        background_tasks.add_task(_cleanup_temp_files, temp_dir, temp_zip.name)

        # Return the ZIP file
        return FileResponse(
            temp_zip.name,
            media_type="application/zip",
            filename="generated_music.zip",
        )

    except subprocess.TimeoutExpired as e:
        logging.error("MusicGen subprocess timed out")
        _cleanup_temp_files(temp_dir, temp_zip.name if temp_zip else None)
        raise HTTPException(status_code=504, detail="Music generation timed out") from e
    except Exception as e:
        logging.error("Music generation error: %s", e)
        _cleanup_temp_files(temp_dir, temp_zip.name if temp_zip else None)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}") from e


def _cleanup_temp_files(temp_dir, temp_zip_path):
    """Clean up temporary files and directory"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info("Cleaned up temp directory: %s", temp_dir)
        except OSError as e:
            logging.warning("Failed to clean up temp directory %s: %s", temp_dir, e)

    if temp_zip_path and os.path.exists(temp_zip_path):
        try:
            os.unlink(temp_zip_path)
            logging.info("Cleaned up temp zip file: %s", temp_zip_path)
        except OSError as e:
            logging.warning("Failed to clean up temp zip %s: %s", temp_zip_path, e)


@router.post("/ttm")
async def text_to_music(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    seed: int = Form(),
    seconds: float = Form(),
):
    """Convert text prompt to music audio files (returns ZIP of MP3 files)"""
    return await asyncio.wait_for(
        process_music_with_subprocess(prompt, seed, seconds, background_tasks),
        timeout=610,
    )
