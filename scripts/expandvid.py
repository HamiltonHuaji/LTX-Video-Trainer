import torch
import torchvision
import argparse

def map_video_frames(input_filename, output_filename):
    video, _, info = torchvision.io.read_video(input_filename, pts_unit='sec')
    output_frames = [video[0]] + [frame for i in range(1, video.shape[0]) for frame in video[i].unsqueeze(0).repeat(8, 1, 1, 1)]
    torchvision.io.write_video(output_filename, torch.stack(output_frames), fps=int(info['video_fps']), options={'crf': '17', 'preset': 'veryslow'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='视频帧映射工具')
    parser.add_argument('--input', '-i', help='输入视频文件名')
    parser.add_argument('--output', '-o', help='输出视频文件名')
    args = parser.parse_args()
    map_video_frames(args.input, args.output)
