from pathlib import Path


OK_EXTENSIONS = {
    'wav',
    'mp4',
    'avi',
    'flv',
    'wbm',
    'ogg'
}


def get_video_list(video_dir: Path):
    return [x for x in video_dir.glob('*.*') if
            x.name.split('.')[-1] in OK_EXTENSIONS]
