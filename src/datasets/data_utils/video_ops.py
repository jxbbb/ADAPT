from PIL import Image
import io
import av
import torch
import numpy as np
from src.datasets.data_utils import video_decoder as decoder
import code

def get_video_decoding_kwargs(container, num_frames, target_fps,
                              num_clips=None, clip_idx=None,
                              sampling_strategy="rand",
                              safeguard_duration=False, video_max_pts=None,
                              start=None, end=None):
    if num_clips is None:
        three_clip_names = ["start", "middle", "end"]  # uniformly 3 clips
        assert sampling_strategy in ["rand", "uniform"] + three_clip_names
        if sampling_strategy == "rand":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=-1,  # random sampling
                num_clips=None,  # will not be used when clip_idx is `-1`
                target_fps=target_fps,
                start=start, end=end
            )
        elif sampling_strategy == "uniform":
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,  # will not be used when clip_idx is `-2`
                num_frames=num_frames,
                clip_idx=-2,  # uniformly sampling from the whole video
                num_clips=1,  # will not be used when clip_idx is `-2`
                target_fps=target_fps,  # will not be used when clip_idx is `-2`
                start=start, end=end
            )
        else:  # in three_clip_names
            decoder_kwargs = dict(
                container=container,
                sampling_rate=1,
                num_frames=num_frames,
                clip_idx=three_clip_names.index(sampling_strategy),
                num_clips=3,
                target_fps=target_fps,
                start=start, end=end
            )
    else:  # multi_clip_ensemble, num_clips and clip_idx are only used here
        assert clip_idx is not None
        # sampling_strategy will not be used, as uniform sampling will be used by default.
        # uniformly sample `num_clips` from the video,
        # each clip sample num_frames frames at target_fps.
        decoder_kwargs = dict(
            container=container,
            sampling_rate=1,
            num_frames=num_frames,
            clip_idx=clip_idx,
            num_clips=num_clips,
            target_fps=target_fps,
            safeguard_duration=safeguard_duration,
            video_max_pts=video_max_pts,
            start=start, end=end
        )
    return decoder_kwargs

def extract_frames_from_video_path(
        video_path, target_fps=3, num_frames=3,
        multi_thread_decode=False, sampling_strategy="rand",
        safeguard_duration=False, start=None, end=None):
    in_mem_bytes_io = video_path
    try:
        frames, video_max_pts = extract_frames_from_video_binary(
            in_mem_bytes_io, target_fps=target_fps, num_frames=num_frames,
            multi_thread_decode=multi_thread_decode,
            sampling_strategy=sampling_strategy,
            safeguard_duration=safeguard_duration,
            start=start, end=end)
    except Exception as e:
        print(f"Error processing video {video_path}, {e}")
        return None, None
    return frames, video_max_pts


def extract_frames_from_video_binary(
        in_mem_bytes_io, target_fps=3, num_frames=3, num_clips=None, clip_idx=None,
        multi_thread_decode=False, sampling_strategy="rand",
        safeguard_duration=False, video_max_pts=None,
        start=None, end=None):
    """
    Args:
        in_mem_bytes_io: binary from read file object
            >>> with open(video_path, "rb") as f:
            >>>     input_bytes = f.read()
            >>> frames = extract_frames_from_video_binary(input_bytes)
            OR from saved binary in lmdb database
            >>> env = lmdb.open("lmdb_dir", readonly=True)
            >>> txn = env.begin()
            >>> stream = io.BytesIO(txn.get(str("key").encode("utf-8")))
            >>> frames = extract_frames_from_video_binary(stream)
            >>> from torchvision.utils import save_image
            >>> save_image(frames[0], "path/to/example.jpg")  # save the extracted frames.
        target_fps: int, the input video may have different fps, convert it to
            the target video fps before frame sampling.
        num_frames: int, number of frames to sample.
        multi_thread_decode: bool, if True, perform multi-thread decoding.
        sampling_strategy: str, how to sample frame from video, one of
            ["rand", "uniform", "start", "middle", "end"]
            `rand`: randomly sample consecutive num_frames from the video at target_fps
                Note it randomly samples a clip containing num_frames at target_fps,
                not uniformly sample from the whole video
            `uniform`: uniformly sample num_frames of equal distance from the video, without
                considering target_fps/sampling_rate, etc. E.g., when sampling_strategy=uniform
                and num_frames=3, it samples 3 frames at [0, N/2-1, N-1] given a video w/ N frames.
                However, note that when num_frames=1, it will sample 1 frame at [0].
                Also note that `target_fps` will not be used under `uniform` sampling strategy.
            `start`/`middle`/`end`: first uniformly segment the video into 3 clips, then sample
                num_frames from the corresponding clip at target_fps. E.g., num_frames=3, a video
                w/ 30 frames, it samples [0, 1, 2]; [9, 10, 11]; [18, 19, 20] for start/middle/end.
            If the total #frames at target_fps in the video/clip is less than num_frames,
            there will be some duplicated frames
        num_clips: int,
        clip_idx: int
        safeguard_duration:
        video_max_pts: resue it to improve efficiency
    Returns:
        torch.uint8, (T, C, H, W)
    """
    try:
        # Add `metadata_errors="ignore"` to ignore metadata decoding error.
        # When verified visually, it does not seem to affect the extracted frames.
        video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
    except Exception as e:
        print(f"extract_frames_from_video_binary(), Exception in loading video binary: {e}")
        return None, None

    if multi_thread_decode:
        # Enable multiple threads for decoding.
        video_container.streams.video[0].thread_type = "AUTO"
    try:
        # (T, H, W, C), channels are RGB
        # see docs in decoder.decode for usage of these parameters.
        decoder_kwargs = get_video_decoding_kwargs(
            container=video_container, num_frames=num_frames,
            target_fps=target_fps, num_clips=num_clips, clip_idx=clip_idx,
            sampling_strategy=sampling_strategy,
            safeguard_duration=safeguard_duration, video_max_pts=video_max_pts, 
            start=start, end=end)
        frames, video_max_pts = decoder.decode(**decoder_kwargs)
    except Exception as e:
        print(f"extract_frames_from_video_binary(), Exception in decoding video: {e}")
        return None, None

    # For some reason in PyAV, the video container may not auto-close, and it could occupy computational resource
    # check more details at https://pyav.org/docs/stable/overview/caveats.html#garbage-collection
    video_container.close()

    # (T, H, W, C) -> (T, C, H, W)
    if frames is not None:
        frames = frames.permute(0, 3, 1, 2)
    return frames, video_max_pts