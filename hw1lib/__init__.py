# -*- coding: utf-8 -*-
"""hw1lib — 3차원비전시스템 HW1 (2·3·4주차 통합) 제공 라이브러리.

제공(완성본) 모듈 — 수정하지 않고 import 해서 사용:
  geometry   : K 생성 · 회전(Rodrigues) · 동차좌표 · 정규화 변환 · 투영
  exif       : 스마트폰 JPEG의 EXIF 읽기 · 촬영 조건표
  board      : 체커보드 코너 검출(OpenCV 허용 범위) · 세트 로딩 · 커버리지 맵
  synth      : 합성 데이터 생성기 (참 K · 왜곡 · 자세를 알고 있는 보드 · 상자 · 맨해튼 장면 · 주행 프레임)
  click_tool : 이미지 위 점 · 선분 클릭 도구 (노트북 안 NotebookClicker, 터미널 CLI 는 대안)
  board_display : 모니터에 띄우는 체커보드 HTML 생성
  io_utils   : 결과 파일 규격(K.json · dist.json · f_table.csv) 저장 · 로드
  checks     : 셀프체크 — 여러분이 작성한 함수를 합성 데이터로 검증

직접 구현해야 하는 부분(TODO)은 모두 notebooks/ 안에 있습니다. 이 패키지의 함수를 바꿔서 TODO를 우회하면
채점에서 해당 항목이 인정되지 않습니다 (TA 재실행 스크립트는 원본 hw1lib 로 실행합니다).
"""
__version__ = '2026.10.1'
