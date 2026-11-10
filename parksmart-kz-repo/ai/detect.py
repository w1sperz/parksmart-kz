"""ParkSmart KZ: определяет занятость парковочных мест на видео с помощью YOLO.

Запуск:
    python ai/detect.py --video data/parking.mp4

Результат (его читает дашборд):
    ai/output/spots.json  - какие места заняты
    ai/output/frame.jpg   - кадр с рамками (зелёные - свободно, красные - занято)
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

# Номера классов в наборе COCO, на котором обучена модель YOLO
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def overlap_ratio(spot_box, det_box):
    """Какая доля площади парковочного места закрыта рамкой машины (от 0 до 1)."""
    sx1, sy1, sx2, sy2 = spot_box
    dx1, dy1, dx2, dy2 = det_box
    w = min(sx2, dx2) - max(sx1, dx1)
    h = min(sy2, dy2) - max(sy1, dy1)
    if w <= 0 or h <= 0:
        return 0.0
    area = (sx2 - sx1) * (sy2 - sy1)
    return (w * h) / area if area > 0 else 0.0


def spot_status(spots, detections, threshold=0.3):
    """Место занято, если какая-то машина закрывает не меньше threshold его площади."""
    result = []
    for spot in spots:
        best = max((overlap_ratio(spot["box"], d) for d in detections), default=0.0)
        result.append({"id": spot["id"], "occupied": best >= threshold})
    return result


def write_json_atomic(path, data):
    """Пишем во временный файл и подменяем, чтобы дашборд не прочитал недописанный JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Определение занятости парковочных мест")
    parser.add_argument("--video", required=True, help="путь к видео или номер камеры (0)")
    parser.add_argument("--spots", default="ai/spots_config.json", help="файл с рамками мест")
    parser.add_argument("--out", default="ai/output", help="папка для результатов")
    parser.add_argument("--model", default="yolov8n.pt", help="модель YOLO (скачается сама)")
    parser.add_argument("--every", type=int, default=15, help="обрабатывать каждый N-й кадр")
    parser.add_argument("--conf", type=float, default=0.35, help="порог уверенности модели")
    parser.add_argument("--threshold", type=float, default=0.3, help="доля места, закрытая машиной")
    parser.add_argument("--delay", type=float, default=0.5, help="пауза между кадрами, секунды")
    parser.add_argument("--loop", action="store_true", help="повторять видео по кругу")
    args = parser.parse_args()

    if not os.path.exists(args.spots):
        raise SystemExit(
            f"Нет файла {args.spots}. Разметьте места командой:\n"
            f"  python ai/mark_spots.py --video {args.video}\n"
            "или скопируйте ai/spots_config.example.json в ai/spots_config.json и поправьте рамки."
        )
    with open(args.spots, encoding="utf-8") as f:
        spots = json.load(f)

    # Тяжёлые библиотеки подключаем здесь, чтобы функции выше можно было проверять без них
    import cv2
    from ultralytics import YOLO

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.model)
    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {args.video}")

    frame_idx = 0
    just_restarted = False
    print("Работаем. Остановить: Ctrl+C")
    while True:
        ok, frame = cap.read()
        if not ok:
            if args.loop and not just_restarted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                just_restarted = True
                continue
            break
        just_restarted = False
        frame_idx += 1
        if frame_idx % args.every:
            continue

        result = model(frame, verbose=False, conf=args.conf)[0]
        detections = []
        for box in result.boxes:
            if int(box.cls[0]) in VEHICLE_CLASSES:
                detections.append(tuple(float(v) for v in box.xyxy[0].tolist()))

        statuses = spot_status(spots, detections, args.threshold)

        for spot, st in zip(spots, statuses):
            x1, y1, x2, y2 = (int(v) for v in spot["box"])
            color = (60, 60, 240) if st["occupied"] else (80, 200, 80)  # цвета в формате BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, spot["id"], (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for x1, y1, x2, y2 in detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

        tmp_img = os.path.join(args.out, "frame.tmp.jpg")
        cv2.imwrite(tmp_img, frame)
        os.replace(tmp_img, os.path.join(args.out, "frame.jpg"))

        occupied = sum(1 for st in statuses if st["occupied"])
        write_json_atomic(os.path.join(args.out, "spots.json"), {
            "updated": datetime.now(timezone.utc).isoformat(),
            "total": len(statuses),
            "occupied": occupied,
            "spots": statuses,
        })
        print(f"кадр {frame_idx}: занято {occupied} из {len(statuses)}")
        time.sleep(args.delay)

    cap.release()
    print("Готово.")


if __name__ == "__main__":
    main()
