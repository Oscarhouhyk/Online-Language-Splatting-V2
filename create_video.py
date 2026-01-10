import cv2
import os
import argparse
import glob
import re
from tqdm import tqdm

def natural_sort_key(s):
    """用于对包含数字的文件名/目录名进行自然排序"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_video(root_dir, class_name, output_path, fps=10, mode="class"):
    image_files = []
    
    if mode == "class":
        # 查找所有包含 composited 图片的子目录
        # 假设目录结构是 root_dir/{step_dir}/composited/{class_name}_0.png
        # 例如: root_dir/20_00003/composited/floor_0.png
        
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # 对子目录进行自然排序，确保帧顺序正确
        subdirs.sort(key=natural_sort_key)
        
        print(f"Searching for images in {root_dir} for class '{class_name}'...")
        
        target_filename = f"{class_name}_0.png"
        
        for d in subdirs:
            img_path = os.path.join(root_dir, d, "composited", target_filename)
            if os.path.exists(img_path):
                image_files.append(img_path)
    
    elif mode == "full_seq":
        print(f"Searching for all images in {root_dir} for full sequence reconstruction...")
        # 查找 root_dir 下的所有图片文件 (假设是 png 或 jpg)
        # 这里的路径通常是 .../pred
        files = os.listdir(root_dir)
        files.sort(key=natural_sort_key)
        
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root_dir, f))
    
    if not image_files:
        print(f"No images found in {root_dir}")
        return

    print(f"Found {len(image_files)} frames.")

    # 读取第一帧以获取尺寸
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"Failed to read first frame: {image_files[0]}")
        return
        
    height, width, layers = first_frame.shape
    size = (width, height)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Writing video to {output_path} with {fps} FPS...")
    
    # 使用 mp4v 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    for filename in tqdm(image_files):
        img = cv2.imread(filename)
        if img is not None:
            # 简单检查大小是否一致，防止视频损坏
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, size)
            out.write(img)
        else:
            print(f"Warning: Could not read image {filename}")
            
    out.release()
    print("Video generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from image frames in subdirectories.")
    parser.add_argument("--root_dir", type=str, required=True, 
                        help="Root directory containing subfolders (e.g., .../eval_results/room0) or images directly")
    parser.add_argument("--class_name", type=str, default=None, 
                        help="Class name to search for (e.g., floor, sofa). Required if mode is 'class'")
    parser.add_argument("--output", type=str, default="output_video.mp4", 
                        help="Path for the output video file")
    parser.add_argument("--fps", type=int, default=10, 
                        help="Frames per second for the video")
    parser.add_argument("--mode", type=str, default="class", choices=["class", "full_seq"],
                        help="Mode: 'class' for composited objects in subdirs, 'full_seq' for flat directory of images")

    args = parser.parse_args()
    
    if args.mode == "class" and not args.class_name:
        parser.error("--class_name is required when mode is 'class'")

    create_video(args.root_dir, args.class_name, args.output, args.fps, args.mode)

# Example usage (Class mode):
# python create_video.py --root_dir .../eval_results/room0 --class_name floor --output floor.mp4

# Example usage (Full Seq mode):
# python create_video.py --root_dir /data/houyj/robotics/online_lang_splatting/results/2-stage/room_0/2026-01-08-12-07-59/psnr/before_opt/pred --mode full_seq --output full_seq.mp4
