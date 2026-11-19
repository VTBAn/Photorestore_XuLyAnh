"""Script tự động tải 3 tệp mô hình tô màu ảnh đen trắng (Zhang et al., 2016)."""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = {
    "colorization_deploy_v2.prototxt": (
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/"
        "colorization_deploy_v2.prototxt"
    ),
    "pts_in_hull.npy": (
        "https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy"
    ),
    "colorization_release_v2.caffemodel": (
        "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"
    ),
}

def main():
    for filename, url in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"[Đã có] {filename}")
            continue
        print(f"[Đang tải] {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  -> Hoàn tất ({os.path.getsize(dest) / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"  -> Lỗi khi tải {filename}: {e}")
            print(f"     Bạn có thể tải thủ công từ: {url}")
            print(f"     và đặt vào thư mục: {MODELS_DIR}")

if __name__ == "__main__":
    main()
