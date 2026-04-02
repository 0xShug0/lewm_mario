from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np

from mario_lewm.fm2 import parse_fm2, read_fm2_header

GD_HEADER_SIZE = 11
FRAME_HEIGHT = 240
FRAME_WIDTH = 256
CHANNELS_ARGB = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay FM2 traces in FCEUX and export per-trace .npz episodes with frames and actions."
    )
    parser.add_argument("--trace-root", type=Path, default=Path("traces"), help="Directory containing .fm2 traces.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory that will receive exported .npz episodes."
    )
    parser.add_argument("--fceux-dir", type=Path, default=Path("fceux"), help="Directory containing fceux64.exe.")
    parser.add_argument("--rom", type=Path, default=Path("fceux") / "SMB.nes", help="ROM file to load in FCEUX.")
    parser.add_argument("--save-every", type=int, default=1, help="Capture every Nth movie frame.")
    parser.add_argument(
        "--capture-initial-frame",
        action="store_true",
        help="Capture the initial observation before the first movie action. Recommended for LeWM-style blocked datasets.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on replayed movie frames. 0 disables.")
    parser.add_argument(
        "--visual-debug",
        action="store_true",
        help="Run FCEUX in visible, throttled mode for manual inspection while still exporting.",
    )
    parser.add_argument(
        "--debug-exit-delay",
        type=int,
        default=120,
        help="Extra frames to keep FCEUX open after export completes when --visual-debug is enabled.",
    )
    parser.add_argument("--keep-staging", action="store_true", help="Keep raw capture files after packing .npz.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz episodes.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of traces to export.")
    return parser.parse_args()


def compute_nes_unheadered_md5_base64(path: Path) -> str:
    data = path.read_bytes()
    trainer = 512 if data[6] & 0x04 else 0
    payload = data[16 + trainer :]
    return base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")


def make_job_lua(
    path: Path,
    *,
    rom_path: Path,
    trace_path: Path,
    capture_path: Path,
    metadata_path: Path,
    save_every: int,
    max_frames: int,
    visual_debug: bool,
    debug_exit_delay: int,
    capture_initial_frame: bool,
) -> None:
    content = "\n".join(
        [
            "return {",
            f"  rom_path = [[{rom_path}]],",
            f"  trace_path = [[{trace_path}]],",
            f"  output_capture_path = [[{capture_path}]],",
            f"  output_metadata_path = [[{metadata_path}]],",
            f"  save_every = {int(save_every)},",
            f"  max_frames = {int(max_frames)},",
            f"  visual_debug = {str(bool(visual_debug)).lower()},",
            f"  debug_exit_delay = {int(debug_exit_delay)},",
            f"  capture_initial_frame = {str(bool(capture_initial_frame)).lower()},",
            "}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def parse_kv_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def decode_gd_capture(raw_bytes: bytes, blob_size: int) -> np.ndarray:
    if blob_size <= GD_HEADER_SIZE:
        raise ValueError(f"Unexpected blob size {blob_size}.")
    if len(raw_bytes) % blob_size != 0:
        raise ValueError("Capture size is not divisible by blob size; raw export looks corrupted.")
    frame_count = len(raw_bytes) // blob_size
    pixel_bytes = FRAME_HEIGHT * FRAME_WIDTH * CHANNELS_ARGB
    if blob_size - GD_HEADER_SIZE != pixel_bytes:
        raise ValueError(
            f"Expected {pixel_bytes} pixel bytes after the GD header, got {blob_size - GD_HEADER_SIZE}."
        )
    frames = np.empty((frame_count, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for frame_idx in range(frame_count):
        start = frame_idx * blob_size + GD_HEADER_SIZE
        stop = start + pixel_bytes
        argb = np.frombuffer(raw_bytes[start:stop], dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, 4)
        frames[frame_idx] = argb[..., 1:4]
    return frames


def build_trace_metadata(trace_path: Path, rom_checksum_base64: str) -> dict[str, str]:
    header = read_fm2_header(trace_path)
    return {
        "trace_name": trace_path.name,
        "fm2_version": header.get("version", ""),
        "emuVersion": header.get("emuVersion", ""),
        "rerecordCount": header.get("rerecordCount", ""),
        "palFlag": header.get("palFlag", ""),
        "romFilename": header.get("romFilename", ""),
        "romChecksum": header.get("romChecksum", ""),
        "guid": header.get("guid", ""),
        "rom_checksum_base64_loaded": rom_checksum_base64,
    }


def pack_episode(
    trace_path: Path,
    output_path: Path,
    capture_path: Path,
    metadata_path: Path,
    rom_checksum_base64: str,
) -> dict[str, int | str]:
    export_meta = parse_kv_metadata(metadata_path)
    blob_size = int(export_meta["blob_size"])
    raw_bytes = capture_path.read_bytes()
    frames = decode_gd_capture(raw_bytes, blob_size)

    actions = parse_fm2(trace_path).astype(np.float32)
    save_every = max(1, int(export_meta.get("save_every", "1")))
    if save_every > 1:
        actions = actions[save_every - 1 :: save_every]

    capture_initial_frame = export_meta.get("capture_initial_frame", "false") == "true"
    if capture_initial_frame:
        # Exact LeWM-style alignment keeps one more frame than action row so we
        # can form (observation_t, action_block_t, observation_t+1) triples later.
        usable = min(len(actions), max(0, len(frames) - 1))
        if usable == 0:
            raise ValueError(f"No usable initial-frame transitions were produced for {trace_path.name}.")
        frame_save = frames[: usable + 1]
        action_save = actions[:usable]
    else:
        usable = min(len(frames), len(actions))
        if usable == 0:
            raise ValueError(f"No usable frame/action pairs were produced for {trace_path.name}.")
        frame_save = frames[:usable]
        action_save = actions[:usable]

    trace_meta = build_trace_metadata(trace_path, rom_checksum_base64)
    metadata = {
        **trace_meta,
        "capture_backend": "fceux",
        "capture_blob_size": blob_size,
        "captured_frames": int(export_meta.get("captured_frames", usable)),
        "last_movie_frame": int(export_meta.get("last_movie_frame", 0)),
        "movie_length": int(export_meta.get("movie_length", 0)),
        "movie_mode_end": export_meta.get("movie_mode_end", ""),
        "save_every": save_every,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "capture_initial_frame": capture_initial_frame,
        "usable_action_steps": int(usable),
    }

    np.savez_compressed(
        output_path,
        frames=frame_save,
        actions=action_save,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return {
        "trace": trace_path.name,
        "episode": output_path.name,
        "frames": int(len(frame_save)),
        "actions": int(len(action_save)),
        "movie_length": int(metadata["movie_length"]),
    }


def launch_fceux(fceux_exe: Path, rom_path: Path, exporter_lua: Path, job_lua: Path, workdir: Path) -> None:
    env = dict(os.environ)
    env["FCEUX_EXPORT_JOB"] = str(job_lua.resolve())
    command = [str(fceux_exe.resolve()), "-lua", str(exporter_lua.resolve()), str(rom_path.resolve())]
    completed = subprocess.run(command, cwd=workdir, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"FCEUX export failed with exit code {completed.returncode}.")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fceux_exe = args.fceux_dir / "fceux64.exe"
    if not fceux_exe.exists():
        raise FileNotFoundError(fceux_exe)
    if not args.rom.exists():
        raise FileNotFoundError(args.rom)

    rom_checksum = compute_nes_unheadered_md5_base64(args.rom)
    exporter_lua = Path("fceux_export_trace.lua")
    if not exporter_lua.exists():
        raise FileNotFoundError(exporter_lua)

    trace_paths = sorted(args.trace_root.glob("*.fm2"))
    if args.limit > 0:
        trace_paths = trace_paths[: args.limit]
    if not trace_paths:
        raise FileNotFoundError(f"No .fm2 files found under {args.trace_root}.")

    staging_dir = args.output_dir / "_fceux_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, int | str]] = []
    for trace_path in trace_paths:
        output_path = args.output_dir / f"{trace_path.stem}.npz"
        if output_path.exists() and not args.overwrite:
            continue

        trace_header = read_fm2_header(trace_path)
        expected_checksum = trace_header.get("romChecksum", "").removeprefix("base64:")
        if expected_checksum and expected_checksum != rom_checksum:
            print(
                json.dumps(
                    {
                        "trace": trace_path.name,
                        "status": "skipped_checksum_mismatch",
                        "trace_checksum": expected_checksum,
                        "rom_checksum": rom_checksum,
                    }
                )
            )
            continue

        trace_id = zlib.crc32(str(trace_path.resolve()).encode("utf-8")) & 0xFFFFFFFF
        capture_path = staging_dir / f"{trace_id:08x}.gdv"
        metadata_path = staging_dir / f"{trace_id:08x}.meta"
        job_lua = staging_dir / f"{trace_id:08x}.job.lua"

        for path in (capture_path, metadata_path, job_lua):
            if path.exists():
                path.unlink()

        make_job_lua(
            job_lua,
            rom_path=args.rom.resolve(),
            trace_path=trace_path.resolve(),
            capture_path=capture_path.resolve(),
            metadata_path=metadata_path.resolve(),
            save_every=max(1, args.save_every),
            max_frames=max(0, args.max_frames),
            visual_debug=args.visual_debug,
            debug_exit_delay=max(0, args.debug_exit_delay),
            capture_initial_frame=args.capture_initial_frame,
        )
        launch_fceux(fceux_exe, args.rom, exporter_lua, job_lua, args.fceux_dir)
        if not capture_path.exists() or not metadata_path.exists():
            raise RuntimeError(f"FCEUX did not write capture outputs for {trace_path.name}.")

        packed = pack_episode(
            trace_path=trace_path,
            output_path=output_path,
            capture_path=capture_path,
            metadata_path=metadata_path,
            rom_checksum_base64=rom_checksum,
        )
        summary.append(packed)
        print(json.dumps(packed))

        if not args.keep_staging:
            capture_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            job_lua.unlink(missing_ok=True)

    (args.output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.keep_staging and staging_dir.exists() and not any(staging_dir.iterdir()):
        shutil.rmtree(staging_dir)


if __name__ == "__main__":
    main()
