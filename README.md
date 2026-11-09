# HW1 — 내 스마트폰 카메라를 네 가지 방법으로 캘리브레이션하기

**배포: 2026-10-20**
개인 과제 · 보고서 PPT 제출(제공된 템플릿은 단순 참고용)
과제 공지: **2026-10-20** · 제출 마감: **2026-11-10 23:59**

2·3·4주차 내용을 통합하여 **본인 스마트폰 사진**에서 같은 카메라의 초점거리 f를 EXIF 예측, Zhang 체커보드, 상자 DLT, 소실점으로 추정합니다. 방법별 결과를 같은 픽셀 해상도에서 비교하고, 오차·불확실성·불일치의 원인을 설명합니다. 정답은 사람마다 다릅니다. 제출처와 상세 안내는 `HW1_과제안내.pdf`를 확인하세요.

## 1. 개발 환경과 첫 실행

1. **Python 3.10–3.12**를 설치합니다.
2. **VS Code**와 Microsoft의 **Python**, **Jupyter** 확장을 설치합니다.
3. 압축을 푼 `HW1` 폴더를 VS Code의 **File → Open Folder**로 엽니다. 이 폴더 안에 `hw1_config.json`, `requirements.txt`, `notebooks/`, `hw1lib/`가 있어야 합니다.
4. `Ctrl/⌘ + Shift + P` → **Python: Create Environment** → **Venv** → 설치한 Python 선택 → **requirements.txt 체크**. 패키지 설치가 끝날 때까지 기다립니다.
5. `notebooks/Part0_촬영_프로토콜.ipynb`를 열고 오른쪽 위 **Select Kernel**에서 이 폴더의 `.venv`를 선택합니다. 공통 설정 셀과 환경 점검 셀을 실행하여 경로와 패키지 버전을 확인합니다.

처음에는 `C:\course\HW1` 또는 `~/course/HW1` 같은 짧은 경로가 편리합니다. 제공 이미지 로더는 한글·공백·대괄호가 있는 경로를 처리합니다. 다만 일부 Jupyter 실행 환경이 대괄호(`[ ]`) 경로를 패턴으로 해석하여 **커널 시작**에 실패하면, HW1 폴더를 짧은 경로로 복사한 뒤 그곳에서 가상환경과 커널을 다시 선택하세요. 한글 파일명을 바꿀 필요는 없습니다.

### 모든 Part가 공유하는 설정

데이터 경로를 노트북마다 고치지 말고, 루트의 **`hw1_config.json` 한 곳**을 바꿉니다. 기본값은 다음과 같습니다.

```json
{"mode": "student", "scale": 1.0}
```

| 설정 | 데이터 | 결과 저장 위치 |
|---|---|---|
| `"mode": "student"` | 본인 사진 `data/` | 제출용 `results/` |
| `"mode": "demo"` | 제공 합성 사진 `data_demo/` | 연습용 `results_demo/` |

- `scale`은 Part 0의 체커보드 코너 검출 배율입니다. 기본 `1.0`, 큰 사진이 느리면 `0.5`를 사용할 수 있습니다. 세트 A·B·C 전체에 같은 값을 적용합니다.
- 모드나 배율을 바꾸면 각 노트북의 커널을 다시 시작하고 **Part 0부터 필요한 Part를 순서대로 재실행**합니다. 이전 결과를 보존하려면 결과 폴더를 별도로 백업하세요.
- 결과 폴더의 `dataset_context.json`은 데이터 출처를 기록합니다. 다른 출처의 결과 또는 v4 결과를 같은 폴더에 합치지 마세요. 출처 불일치 오류가 나면 기존 결과 폴더를 다른 이름으로 보관하고 Part 0부터 다시 생성합니다.
- 새 결과 경로가 필요하면 설정 파일에 해당 모드의 경로를 지정할 수 있습니다. 예: `"demo": {"data": "data_demo", "results": "results_demo_half"}`. student/demo의 데이터·결과 경로 네 개는 서로 달라야 합니다.

합성 데이터는 배포본에 포함되어 있습니다. 없으면 Part 0의 **합성 데모 데이터** 셀에서 생성할 수 있습니다. 참값은 `data_demo/_truth.json`에 있습니다. 데모 모드에서도 학생 TODO는 직접 구현해야 합니다. **제출은 student 모드에서 본인 사진으로 실행한 결과**입니다.

### 실행 순서와 제공 코드 검사

**Part 0 → Part 1 → Part 2 → Part 1의 1.3·1.4 재실행 → Part 3 → Part 4 → Part 5 → Part 6** 순서로 진행합니다. Part 1의 일부 비교·삽입 실험은 Part 2의 K·왜곡 계수가 나온 뒤 완성합니다. 각 Part에서는 설명을 읽고 TODO를 채운 다음 실행합니다. 미구현 TODO의 `NotImplementedError`는 과제용 빈 함수임을 알리는 표시입니다.

VS Code 터미널에서 `.venv`가 선택된 상태로, HW1 폴더에서 다음을 실행하면 제공 라이브러리와 합성 데이터가 동작하는지 검사할 수 있습니다. 이 검사는 학생 TODO의 채점이나 완성을 대신하지 않습니다.

```bash
python tests/run_tests.py
```

OpenCV는 **각 Part 안내에서 허용한 함수만** 사용합니다. NumPy·SciPy·Matplotlib은 자유롭게 사용할 수 있습니다. 코너 검출, `findHomography`, `Rodrigues`, 일부 클릭점의 `undistortPoints`, 검증용 비교 등의 허용 범위는 해당 노트북을 따릅니다.

## 2. 구성과 구현 범위

| 폴더 / 파일 | 내용 |
|---|---|
| `notebooks/Part0_촬영_프로토콜.ipynb` | 촬영 체크리스트·세트 A/B/C·실측 기록·EXIF 표·코너 검출·커버리지 맵 |
| `notebooks/Part1_사양에서_f_예측.ipynb` | EXIF→f, H 분해로 거리, K 변환 규칙, 이미지 삽입 |
| `notebooks/Part2_Zhang_캘리브레이션.ipynb` | 닫힌 해→k 초기화→LM, 합성 검증, OpenCV 비교, 세트 A/B/C |
| `notebooks/Part3_상자_DLT.ipynb` | 정규화 DLT→P→RQ 분해, 퇴화 |
| `notebooks/Part4_소실점_단일시점.ipynb` | 소실점→주점·f, 높이 측정, 정면 사진, 주행 프레임 피치 |
| `notebooks/Part5_왜곡_보정.ipynb` | 역방향 언디스토션·plumb-line·κ↔k 환산 함정 |
| `notebooks/Part6_종합_함정문항.ipynb` | f 비교표·함정 5문항·LLM 풀이 감정 |
| `HW1_과제안내.pdf` | 과제 목적·일정·제출·평가 안내 |
| `HW1_보고서_템플릿.pptx` | 제출 보고서 양식 |
| `hw1lib/` | 완성된 제공 라이브러리 |
| `hw1_config.json` | 모든 Part의 데이터 모드·코너 검출 배율 설정 |
| `checkerboard_screen.html` | 모니터 체커보드: 10×7칸, 내부 코너 9×6. 전체 화면과 모니터 대각선으로 칸 크기(mm) 확인 |
| `data/`, `results/` | 본인 사진과 제출 결과. 결과 규격은 `hw1lib/io_utils.py` 머리말 참고 |
| `data_demo/`, `results_demo/` | 합성 연습 데이터와 연습 결과. 제출 결과와 분리 |

**TODO 함수는 총 32개(Part 간 복사 2개 포함)**입니다. 각 TODO에는 MVG 2판·Zhang 2000의 절·식 번호와 단계별 힌트가 있습니다. 해당 강의 내용을 찾아 함께 공부하고, `hw1lib.checks`로 합성 데이터에서 구현을 확인하세요.

Part 2의 LM 기본 모델은 **skew=0 고정**, `fx, fy, cx, cy`와 방사 왜곡 **k₁·k₂**를 추정합니다. 접선 왜곡 `p₁=p₂=0`, `k₃=0`이며 `fx`와 `fy`는 각각 추정합니다. OpenCV의 같은 모델과 비교한 뒤, 추가로 5계수 모델의 차이를 분석합니다. 모델과 종료 조건이 다르거나 잡음이 있으면 수치가 완전히 같을 필요는 없습니다.

### Part 2 구현 실패 시 진행 방법

뒤 Part를 진행하기 위한 OpenCV 대체 경로는 scratch 구현과 독립적입니다. **Part 0 완료 → Part 2 공통 설정 → 2.7 OpenCV 비교 셀 → 마지막 Fallback 셀** 순서로 실행하고, Fallback 셀의 `USE_OPENCV_FALLBACK = True`로 바꿉니다. 이 경로가 `K.json`·`dist.json`을 저장하므로 Part 3–6을 진행할 수 있습니다.

TODO를 임의로 OpenCV 호출로 대체하는 것은 허용되지 않습니다.

## 3. 사진·실측·좌표 규격

```text
data/setA/*.jpg        정면 평행 15장
data/setB/*.jpg        중앙 집중 15장
data/setC/*.jpg        권장 프로토콜 15장
data/box/box1.jpg      세 면·코너 7개 + box1_points.json
data/box/box2_oneface.jpg       한 면 + box2_oneface_points.json
data/building/b1.jpg           + b1_lines.json, b1_plumb.json, b1_height_points.json
data/building/b2_frontal.jpg   + b2_frontal_lines.json
data/driving/frame.png        제공 프레임 + frame_K.json, frame_lines.json
data/tape/*.jpg        줄자 실측 사진
data/texture.png       삽입할 이미지
data/fisheye/          어안 보너스 데이터 (선택)
data/noise/            RAW·JPEG 보너스 데이터 (선택)
```

체커보드 세트의 사진 파일명은 자유입니다. 상자·건물·주행 사진은 위 이름을 사용합니다. 클릭 JSON은 Part 3·4·5에서 만듭니다. 제공 주행 프레임의 `frame_lines.json`도 Part 4에서 직접 클릭하여 생성합니다. 왼쪽 클릭은 점 추가, 오른쪽 클릭은 취소, 가운데 클릭은 현재 방향·에지 종료입니다. 툴바 돋보기로 확대한 뒤 **돋보기 모드를 해제하고** 클릭하세요.

Part 0의 실측 기록에서 다음 두 파일명은 반드시 실제로 측정한 `setC` 사진과 일치해야 합니다.

- `board_distance_image`: 렌즈에서 보드 중심까지 거리를 잰 사진. `board_distance_mm`와 한 쌍입니다.
- `camera_height_image`: 보드 평면에서 렌즈까지 수직 거리를 잰 사진. `camera_height_mm`와 한 쌍입니다. 위 사진과 달라도 됩니다.

두 파일명을 코너 검출 실패 목록과 함께 확인하세요. 실측 사진의 코너가 없으면 해당 실험은 오류로 멈추며, 다른 사진으로 자동 대체하지 않습니다. `data/measurements.json`이 있으면 Part 0이 셀의 예시 dict 대신 그 파일을 읽으므로, 파일을 수정한 뒤 셀을 다시 실행하세요. 최종 실측 기록은 `results/measurements.json`에 저장합니다.

### 픽셀 좌표와 해상도

제공 로더는 **EXIF 자동 회전을 적용하지 않고 저장된 픽셀 격자**를 읽습니다. 휴대폰 사진 앱의 표시 방향과 다르게 보일 수 있습니다. 픽셀 중심은 정수 좌표이며 원점은 좌상단입니다. Orientation이 서로 다른 사진을 섞지 마세요.

원본 사진을 리사이즈·회전·크롭하여 저장하지 마세요. 계산량을 줄일 때는 설정의 `scale`을 사용합니다. Part 2는 K와 함께 `image_size=[W,H]`를 저장합니다. Part 3–5는 이 크기를 이용해 K를 원본 사진의 픽셀 격자로 변환하고, Part 6은 모든 f·주점·σ(f)를 `setC` 원본 크기로 맞춥니다. 원본 크기는 `f_table_context.json`에 기록합니다. **정규화 좌표에서 정의된 왜곡 계수는 해상도가 바뀌어도 나누거나 곱하지 않습니다.** 화면비·방향·시야가 달라진 사진에는 단순 해상도 변환을 적용할 수 없습니다.

## 3-1. 촬영 전 확인

- **공통**: 1× 메인 렌즈, 같은 해상도·가로 방향, 디지털 줌 금지. 가능한 기종에서는 HDR·장면 최적화·렌즈 보정·자동 렌즈 전환을 끄고 초점·노출을 잠급니다. 설정 메뉴는 기종별로 다르므로 실제 사용 조건을 기록합니다.
- **iPhone**: 높은 호환성(JPEG), 같은 사진 해상도. 촬영 화면에서 AE/AF 잠금. 렌즈 보정·HDR·매크로 관련 설정은 Part 0의 표에서 확인합니다.
- **Android**: 4:3 및 같은 해상도. 프로 모드에서 ISO·셔터·초점을 고정합니다. 제공 앱에서 고정할 수 있는 설정과 제한을 조건표에 적습니다.
- **체커보드**: 24인치 이상 모니터 권장. 네이티브 해상도, `checkerboard_screen.html` 전체 화면, 대각선 입력 후 칸 크기(mm) 기록. 아래 1000px 눈금자를 자로 확인하고 H로 패널을 숨깁니다. 카메라를 움직여 자세를 바꿉니다.
- **건물·상자**: `data_demo/`의 해당 사진을 참고합니다. 건물 사진에는 직교하는 세 방향과 같은 바닥 위의 기준·대상 높이가 필요합니다. 상자는 세 면과 코너 7개가 보여야 합니다.

## 4. 제출물

1. **보고서 PPT**: `HW1_보고서_템플릿.pptx` 참고 (자유양식).
2. **코드와 설정**: 실행 결과를 저장한 `notebooks/*.ipynb`, `hw1lib/`, `requirements.txt`, `hw1_config.json`.
3. **실행 결과**: `results/` 전체. `dataset_context.json`과 `measurements.json`도 포함합니다.
4. **원본 사진과 클릭 JSON**: `data/` 전체. EXIF를 유지하고, 원본을 리사이즈하지 않습니다.

`mode="student"`에서 실행한 파일을 **`HW1_학번_이름.zip`**으로 압축합니다. `.venv/`, 캐시, `data_demo/`, `results_demo/`는 제출 압축에서 제외합니다. 제출 마감은 **2026-11-10 23:59**이며, LMS 과제란에 압축 파일을 제출합니다.

## 5. 채점

**기본 100점 + 보너스 10점**: Part 0 5점, Part 1 10점, Part 2 30점, Part 3 15점, Part 4 15점, Part 5 15점, Part 6 10점. 보너스는 Part 5와 Part 6에 각각 5점입니다.

TA는 사진·설정·`results/`로 재실행하여 결과의 일치와 설명을 확인합니다. 같은 코너·같은 모델·같은 설정에서 재현되는지 확인하고, 수치 차이가 있으면 라이브러리 버전·종료 조건·데이터 처리 차이를 보고서에 기록하세요.
