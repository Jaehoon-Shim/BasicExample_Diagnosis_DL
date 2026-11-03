FNN 3상 전류 고장 진단 GUI - 사용 방법

1. Colab 노트북의 학습 셀을 순서대로 실행합니다.
2. 마지막 패키지 생성 셀에서 다운로드한 ZIP을 압축 해제합니다.
3. 아래 3개 파일이 같은 폴더에 있는지 확인합니다.
   - diagnosis_gui.py
   - build_exe.bat
   - fnn_fault_checkpoint.pt
4. Windows에서 build_exe.bat를 더블클릭합니다.
5. 빌드가 끝나면 dist\FNN_Diagnosis_GUI.exe를 실행합니다.

GUI 입력 형식
- Ia 전류 200개 → Ib 전류 200개 → Ic 전류 200개 순서, 총 600개
- 쉼표, 공백 또는 줄바꿈으로 구분해 직접 붙여넣을 수 있습니다.
- CSV/TXT 불러오기도 지원합니다.
- 601개인 행은 마지막 Fault ID 열을 자동으로 제외합니다.

참고
- Python 3.10 또는 3.11(64-bit) 사용을 권장합니다.
- 최초 빌드 시 PyTorch 등을 설치하므로 시간이 걸리고 인터넷 연결이 필요합니다.
- PyTorch를 포함하는 단일 EXE이므로 파일 크기가 큽니다.
- PT를 교체하고 싶으면 새 fnn_fault_checkpoint.pt를 EXE와 같은 폴더에 두면
  내장 PT보다 해당 외부 PT를 우선 사용합니다.
