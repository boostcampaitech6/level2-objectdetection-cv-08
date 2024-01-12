import cv2
import os
import json

prj_dir = '../data/recycle/'

def load_json_file(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def get_annotations_for_image(json_data, image_filename):
    image_info = next((img for img in json_data['images'] if img['file_name'] == image_filename), None)
    if image_info:
        image_id = image_info['id']
        annotations = [anno for anno in json_data['annotations'] if anno['image_id'] == image_id]
        return annotations
    else:
        return None

def display_image_with_annotations(image, annotations, output_path):
    # 이미지 복사
    image_with_annotations = image.copy()

    # 어노테이션을 이미지에 표시
    if annotations:
        for annotation in annotations:
            bbox = annotation['bbox']
            tl_x, tl_y, br_x, br_y = map(int, bbox)
            cv2.rectangle(image_with_annotations, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 2)

    # 이미지 저장
    cv2.imwrite(output_path, image_with_annotations)

# 저장할 이미지 경로
output_img1_path = 'output_image1.jpg'
output_img2_path = 'output_image2.jpg'

json_data = load_json_file(os.path.join(prj_dir, 'train_eye_eda.json'))

# 이미지 파일 경로
img1_path = '../data/recycle/train/4880.jpg'
img2_path = '../data/recycle/train/4882.jpg'

# 이미지 로드
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# 이미지가 None이 아니고 shape 속성이 있는지 확인
if img1 is not None and img2 is not None:
    print("Image 1 shape:", img1.shape)
    print("Image 2 shape:", img2.shape)

    # 어노테이션 정보 가져오기
    annotations_img1 = get_annotations_for_image(json_data, os.path.basename(img1_path))
    annotations_img2 = get_annotations_for_image(json_data, os.path.basename(img2_path))

    # 어노테이션 정보 출력
    print("Annotations for Image 1:", annotations_img1)
    print("Annotations for Image 2:", annotations_img2)

    # 이미지에 어노테이션 표시 및 저장
    display_image_with_annotations(img1, annotations_img1, output_img1_path)
    display_image_with_annotations(img2, annotations_img2, output_img2_path)

    print(f"Saved annotated images to: {output_img1_path} and {output_img2_path}")
else:
    print("One or both images could not be loaded.")
