import os
import sys
import random
import numpy as np
import cv2
from collections import deque
import json
from tqdm import tqdm 

prj_dir = '/data/ephemeral/home/level2-objectdetection-cv-08/data/recycle/'

def load_json_file(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    return data

'''
특정 클래스에 속하는 연결된 영역의 bbox 찾는 역할. 

cls : label / pt : 현재좌표 (x,y)

1. pt에서 시작하여 주변의 픽셀을 DFS로 순회. 
2. 인접 픽셀 (상하좌우)로 순회하면서 해당 픽셀이 cls에 속하고, 방문하지 않았다면
방문 처리하고 좌표 업데이트. 
3. 최종적으로 찾아진 bbox 좌표 반환 
'''
def dfs(cls, pt, visited, img, bbox):
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # 상,하,좌,우로 살펴봄
    q = deque([pt])
    tl_x, tl_y, br_x, br_y = bbox  # 좌상단 x, 좌상단 y, 우하단 x, 우하단 y

    while q:
        pt = q.pop()

        for d in dir:
            x, y = pt[0] + d[0], pt[1] + d[1]
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                if img[x, y] == cls and visited[x][y] == 0:
                    visited[x][y] = 1
                    tl_x = min(tl_x, x)
                    tl_y = min(tl_y, y)
                    br_x = max(br_x, x)
                    br_y = max(br_y, y)
                    q.append([x, y])
                else:
                    visited[x][y] = 1
    # 홀수 값을 짝수로 바꿔줌
    if tl_x % 2 != 0:
        tl_x += 1
    if tl_y % 2 != 0:
        tl_y += 1
    if br_x % 2 != 0:
        br_x += 1
    if br_y % 2 != 0:
        br_y += 1
    return tl_x, tl_y, br_x, br_y
'''
이미지에서 특정 cls 속하는 모든 연결된 영역의 바운딩 박스를 찾아서 반환. 

cls_bbox : 찾은 bbox 박스 저장

1. 배열들 초기화 하고 이미지 순회하면서 cls에 속하는 픽셀을 찾음.
2. 픽셀 기준으로 dfs 함수 호출하여 좌표를 받음. 
3. 좌표들을 다 받아 cls_bbox 리스트에 정렬. 
'''
def find_cls_bbox(cls, img):
    cls_bbox = []
    visited = [[0 for _ in range(img.shape[1])] for _ in range(img.shape[0])]
    tl_x, tl_y = sys.maxsize, sys.maxsize
    br_x, br_y = 0, 0

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] == cls and visited[i][j] == 0:
                visited[i][j] = 1
                tl_x, tl_y, br_x, br_y = dfs(cls, [i, j], visited, img, [tl_x, tl_y, br_x, br_y])
                cls_bbox.append([tl_x, tl_y, br_x, br_y, img[tl_x:br_x, tl_y:br_y].sum()])
                tl_x, tl_y = sys.maxsize, sys.maxsize
                br_x, br_y = 0, 0
            else:
                visited[i][j] = 1

    cls_bbox = sorted(cls_bbox, key=lambda x: x[-1])
    return cls_bbox

'''
Cutmix

img1, img2 : 원본 이미지들 / bbox1 : img1의 bbox list / label1, label2 : annotation /
label_info : left or right 로 어느 쪽에 annotation 저장할지 결정 /
save_path/cutmix : image 저장 경로 / save_path : annotation 저장 경로 /
ith : 현재 image index

'''
def cutmix(img1, img2, bbox1, label1, label2, label_info, save_path, filename, ith):
    # 이미지의 높이, 너비, 채널 수를 구함
    h, w, c = img1.shape
    # 이미지를 4x4 패치로 나누기 위한 패치의 길이 계산
    patch_l = h // 4

    # CutMix 연산을 수행할 패치의 인덱스 생성 및 랜덤하게 섞음
    index = []
    for i in range(4):
        for j in range(4):
            index.append([i, j])
    random.shuffle(index)

    # 랜덤하게 섞인 인덱스에 따라 CutMix 연산 수행
    for [i, j] in index:
        # 패치의 좌표 계산 
        # tl_x, tl_y 패치 좌상단 (x, y)
        # br_x, br_y 패치 우하단 (x, y)
        tl_x, tl_y = patch_l * i, patch_l * j
        br_x, br_y = tl_x + patch_l, tl_y + patch_l
        half_patch_l = patch_l // 2

        # 대체 영역이 백그라운드인 경우에만 수행
        if label2[tl_x:br_x, tl_y:br_y].sum() == 0:
            # 랜덤하게 선택된 이미지1의 바운딩 박스 정보
            rand_tlx, rand_tly, rand_brx, rand_bry, _ = bbox1[-1]
            rand_cx, rand_cy = (rand_tlx + rand_brx) // 2, (rand_tly + rand_bry) // 2

            # 새로운 어노테이션의 좌표 계산
            label_tlx, label_tly, label_brx, label_bry = rand_cx - half_patch_l, rand_cy - half_patch_l, rand_cx + half_patch_l, rand_cy + half_patch_l

            # 좌표가 이미지 범위를 벗어나는 경우 보정
            if label_tlx < 0:
                label_tlx = 0
                label_brx = patch_l
            if label_tly < 0:
                label_tly = 0
                label_bry = patch_l
            if label_brx > h:
                label_brx = h
                label_tlx = h - patch_l
            if label_bry > w:
                label_bry = w
                label_tly = w - patch_l

            # CutMix 연산을 수행하여 이미지2를 갱신
            if tl_y <= w // 2:
                img2[tl_x:br_x, tl_y:br_y] = img1[tl_x:br_x, tl_y:br_y]
                img2[tl_x:br_x, w // 2 + tl_y:w // 2 + br_y] = img1[tl_x:br_x, w // 2 + tl_y:w // 2 + br_y]
            else:
                img2[tl_x:br_x, tl_y - w // 2:br_y - w // 2] = img1[tl_x:br_x, tl_y - w // 2:br_y - w // 2]
                img2[tl_x:br_x, tl_y:br_y] = img1[tl_x:br_x, tl_y:br_y]

            # CutMix 연산을 수행하여 어노테이션2를 갱신
            if label_info == 'left':
                if tl_y < w // 2:
                    label2[tl_x:br_x, tl_y: br_y] = label1[label_tlx:label_brx, label_tly:label_bry]
                else:
                    label2[tl_x:br_x, tl_y - w // 2: br_y - w // 2] = label1[label_tlx:label_brx, label_tly:label_bry]
            else:
                if tl_y < w // 2:
                    label2[tl_x:br_x, tl_y + w // 2: br_y + w // 2] = label1[label_tlx:label_brx, label_tly:label_bry]
                else:
                    label2[tl_x:br_x, tl_y: br_y] = label1[label_tlx:label_brx, label_tly:label_bry]

            # 결과 이미지와 어노테이션을 파일로 저장
            result_filename = f"{ith:04d}.jpg"
            cv2.imwrite(os.path.join(save_path, 'cutmix', result_filename), img2)

            # JSON 파일에 이미지 및 어노테이션 정보 저장
            json_info = {
                "cutmix_info": {
                    "filename": result_filename,
                    "tl_x": tl_x,
                    "tl_y": tl_y,
                    "br_x": br_x,
                    "br_y": br_y,
                    "label_info": label_info
                }
            }
            with open(os.path.join(save_path, f"{result_filename}.json"), 'w') as json_file:
                json.dump(json_info, json_file)

if __name__ == '__main__':
    json_data = load_json_file(os.path.join(prj_dir, 'train_eye_eda.json'))

    image_dir = os.path.join(prj_dir, 'train')  # 이미지 파일이 있는 디렉토리 경로
    labels = [anno for anno in json_data['annotations']]
    
    diff = 0
    for ith, image_info in tqdm(enumerate(json_data['images']), total=len(json_data['images'])):
        image_filename = image_info['file_name']
        image_path = os.path.join(image_dir, image_filename)
        label = [anno for anno in labels if anno['image_id'] == image_info['id']]

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            diff += 1
            continue

        # 이미지 파일이 존재하는 경우에만 계속 진행
        img1 = cv2.imread(image_path)
        if img1.shape != (1024, 1024, 3):
            print(f"Invalid image shape: {img1.shape}")
            diff += 1
            continue

        os.makedirs(os.path.join(prj_dir, 'cutmix'), exist_ok=True)

        # 이미지 경로를 생성할 때 train/ 부분을 이미 지정했기 때문에 두 번 들어가지 않도록 수정
        ith2 = random.randint(0, len(json_data['images']) - 1)
        image_info2 = json_data['images'][ith2]
        image_filename2 = image_info2['file_name']
        image_path2 = os.path.join(image_dir, image_filename2)
        label2 = [anno for anno in labels if anno['image_id'] == image_info2['id']]

        if not os.path.exists(image_path2):
            print(f"Image not found: {image_path2}")
            diff += 1
            continue

        # 이미지 파일이 존재하는 경우에만 계속 진행
        img2 = cv2.imread(image_path2)
        if img2.shape != (1024, 1024, 3):
            print(f"Invalid image shape: {img2.shape}")
            diff += 1
            continue

        print(f'ith: {ith}, ith2: {ith2}')
        print(f'label: {label}')
        print(f'label2: {label2}')

        cutmix(img1, img2, label, label2, prj_dir, 'cutmix', ith)

    print(f'Different shape count: {diff}')