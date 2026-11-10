"""Разметка парковочных мест: рисуете рамки мышью на первом кадре видео.

Запуск:
    python ai/mark_spots.py --video data/parking.mp4

Как пользоваться:
  1. Выделите мышью первое место и нажмите Enter (или пробел).
  2. Так же выделите остальные места. Порядок важен: сначала весь ряд A слева направо,
     потом ряд B и так далее, по 5 мест в ряду (как в дашборде: A-01 ... D-05).
  3. Когда закончите, нажмите Esc.
"""
import argparse
import json

ROWS = "ABCD"
PER_ROW = 5


def make_id(index):
    return f"{ROWS[index // PER_ROW]}-{index % PER_ROW + 1:02d}"


def main():
    parser = argparse.ArgumentParser(description="Разметка парковочных мест на кадре")
    parser.add_argument("--video", required=True, help="путь к видео или картинке")
    parser.add_argument("--out", default="ai/spots_config.json", help="куда сохранить рамки")
    args = parser.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = cv2.imread(args.video)
    if frame is None:
        raise SystemExit(f"Не удалось прочитать: {args.video}")

    boxes = cv2.selectROIs("Места: Enter - принять рамку, Esc - закончить", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if len(boxes) == 0:
        raise SystemExit("Ни одного места не отмечено.")

    limit = len(ROWS) * PER_ROW
    if len(boxes) > limit:
        print(f"Отмечено {len(boxes)} мест, дашборд показывает {limit}. Лишние отброшены.")
        boxes = boxes[:limit]

    spots = [
        {"id": make_id(i), "box": [int(x), int(y), int(x + w), int(y + h)]}
        for i, (x, y, w, h) in enumerate(boxes)
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(spots, f, ensure_ascii=False, indent=2)
    print(f"Сохранено мест: {len(spots)} -> {args.out}")


if __name__ == "__main__":
    main()
