import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import traceback

def test_celeba():
    print("--- Bắt đầu tải và kiểm tra dataset CelebA ---")
    try:
        from datasets import load_dataset
        
        # Tải 100 ảnh đầu tiên
        print("Đang tải dữ liệu CelebA (100 ảnh đầu)...")
        dataset = load_dataset("flwrlabs/celeba", split="train[:100]", trust_remote_code=True)
        print(f"Tải thành công! Số lượng ảnh: {len(dataset)}")
        
        # Tạo thư mục lưu ảnh
        os.makedirs("celeba_samples", exist_ok=True)
        
        # Lặp qua để lưu cả 20 ảnh
        for i in range(len(dataset)):
            sample = dataset[i]
            pil_image = sample['image']
            rgb_array = np.array(pil_image)
            
            if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            else:
                bgr_array = rgb_array
                
            save_path = f"celeba_samples/sample_celeba_{i+1:02d}.jpg"
            cv2.imwrite(save_path, bgr_array)
            
        print(f"Đã lưu thành công toàn bộ {len(dataset)} ảnh CelebA vào thư mục 'celeba_samples'!")
        print(f"Định dạng mảng CelebA (BGR) của ảnh đầu tiên: {bgr_array.shape}")
        
    except Exception as e:
        print("[LỖI] Đã xảy ra lỗi khi tải CelebA:")
        traceback.print_exc()


def test_lfw():
    print("\n--- Bắt đầu tải và kiểm tra dataset LFW ---")
    try:
        from sklearn.datasets import fetch_lfw_people
        
        print("Đang tải dữ liệu LFW...")
        lfw_dataset = fetch_lfw_people(color=True, resize=1.0)
        
        print(f"Tải thành công! Tổng số ảnh LFW: {len(lfw_dataset.images)}")
        
        # Tạo thư mục lưu ảnh
        os.makedirs("lfw_samples", exist_ok=True)
        
        # Lấy thử 100 ảnh đầu tiên của tập LFW để lưu
        num_samples = min(100, len(lfw_dataset.images))
        for i in range(num_samples):
            float_image = lfw_dataset.images[i]
            
            if float_image.max() <= 1.0:
                uint8_image = (float_image * 255).astype(np.uint8)
            else:
                uint8_image = float_image.astype(np.uint8)
                
            if len(uint8_image.shape) == 3 and uint8_image.shape[2] == 3:
                bgr_array = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)
            else:
                bgr_array = uint8_image
            
            save_path = f"lfw_samples/sample_lfw_{i+1:02d}.jpg"
            cv2.imwrite(save_path, bgr_array)
            
        print(f"Đã lưu thành công {num_samples} ảnh LFW vào thư mục 'lfw_samples'!")
        print(f"Định dạng mảng LFW (BGR) của ảnh đầu tiên: {bgr_array.shape}")

    except Exception as e:
        print("[LỖI] Đã xảy ra lỗi khi tải LFW:")
        traceback.print_exc()

if __name__ == "__main__":
    test_celeba()
    test_lfw()
    print("\nQuá trình kiểm tra hoàn tất!")
