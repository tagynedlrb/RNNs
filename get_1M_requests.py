#!/usr/bin/env python3
import os

# =====================================================
# ===== 설정 영역 (처리할 trace 파일 이름 목록) =====
# =====================================================
TRACE_FILES = [
    "trace_10",
    "trace_283",
    "trace_372",
    "trace_661",
    # 필요하면 여기 추가
]

# =====================================================
# ===== 파라미터 =====
# =====================================================
INPUT_DIR = "./alibaba_trace"
OUTPUT_DIR = "./1m_requests_R"
MAX_REQUESTS = 1_000_000  # R 요청 최대 개수

# =====================================================
# ===== 함수 정의 =====
# =====================================================
def extract_R_requests(input_file, output_file, max_requests):
    count = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            parts = line.strip().split(",")  # CSV 포맷 기준
            if len(parts) < 2:
                continue
            opcode = parts[1]
            if opcode == "R":
                fout.write(line)
                count += 1
                if count >= max_requests:
                    break

    print(f"[DONE] {count} 'R' requests saved to {output_file}")


# =====================================================
# ===== 메인 =====
# =====================================================
def main():
    for fname in TRACE_FILES:
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)

        if not os.path.exists(input_path):
            print(f"[WARN] Input file not found: {input_path}, skipping")
            continue

        print(f"[INFO] Processing {input_path} ...")
        extract_R_requests(input_path, output_path, MAX_REQUESTS)


if __name__ == "__main__":
    main()
