import logging
import subprocess
import tempfile
import os
import sys
import shutil
from pathlib import Path
from zipfile import ZipFile

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
        logging.info(f"Created temp directory: {temp_dir}")

        # Run MusicGen processor as subprocess
        musicgen_processor_path = os.path.join(
            os.path.dirname(__file__), "musicgen_processor.py"
        )
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

        logging.info(f"Running MusicGen subprocess: {' '.join(cmd)}")

        # Run subprocess without capturing stdout so logs appear in console
        result = subprocess.run(
            cmd,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,  # 10 minute timeout for music generation
        )

        if result.returncode != 0:
            logging.error(f"MusicGen subprocess failed: {result.stderr}")
            raise HTTPException(
                status_code=500, detail="Music generation failed"
            )

        # Log any stderr output even on success (warnings, etc.)
        if result.stderr:
            logging.info(f"MusicGen subprocess stderr: {result.stderr}")

        # Check if any files were created
        output_files = list(Path(temp_dir).glob("*.mp3"))
        if not output_files:
            raise HTTPException(
                status_code=500, detail="No music files were generated"
            )

        logging.info(f"Generated {len(output_files)} music files")

        # Create ZIP file with all generated music
        temp_zip = tempfile.NamedTemporaryFile(
            delete=False, suffix=".zip", prefix="music_"
        )
        temp_zip.close()

        with ZipFile(temp_zip.name, "w") as zipf:
            for music_file in output_files:
                zipf.write(music_file, music_file.name)
                logging.info(f"Added to zip: {music_file.name}")

        logging.info(f"Created ZIP file: {temp_zip.name}")

        # Add cleanup task to background
        background_tasks.add_task(_cleanup_temp_files, temp_dir, temp_zip.name)

        # Return the ZIP file
        return FileResponse(
            temp_zip.name,
            media_type="application/zip",
            filename="generated_music.zip",
        )

    except subprocess.TimeoutExpired:
        logging.error("MusicGen subprocess timed out")
        _cleanup_temp_files(temp_dir, temp_zip.name if temp_zip else None)
        raise HTTPException(
            status_code=504, detail="Music generation timed out"
        )
    except Exception as e:
        logging.error(f"Music generation error: {e}")
        _cleanup_temp_files(temp_dir, temp_zip.name if temp_zip else None)
        raise HTTPException(
            status_code=500, detail=f"Processing failed: {str(e)}"
        )


def _cleanup_temp_files(temp_dir, temp_zip_path):
    """Clean up temporary files and directory"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logging.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

    if temp_zip_path and os.path.exists(temp_zip_path):
        try:
            os.unlink(temp_zip_path)
            logging.info(f"Cleaned up temp zip file: {temp_zip_path}")
        except Exception as e:
            logging.warning(f"Failed to clean up temp zip {temp_zip_path}: {e}")


@router.post("/ttm")
async def text_to_music(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    seed: int = Form(42),
    seconds: float = Form(25.0),
):
    """Convert text prompt to music audio files (returns ZIP of MP3 files)"""
    return await process_music_with_subprocess(
        prompt, seed, seconds, background_tasks
    )