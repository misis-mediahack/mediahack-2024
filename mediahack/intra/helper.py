from pathlib import Path


def get_video_list(video_dir: Path):
    return [x for x in video_dir.glob('*.*') if
            x not in {'dashboard_data.csv', 'readme.md', 'segment_dict.xlsx', 'train_segments.csv'}]
