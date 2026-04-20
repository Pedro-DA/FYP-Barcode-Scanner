import numpy as np

class telemetry:
    def __init__(self):
        self.latenciesMs = []
        self.detections = []  # list of {'label': str, 'decoded': bool}

    def recordFrame(self, latencyMs, frameDetections):
        # frameDetections: list of (label, bbox, conf, angle)
        # decoded: parallel list of str|None from parseDecodeString
        self.latenciesMs.append(latencyMs)
        for label, _, conf, _ in frameDetections:
            self.detections.append({'label': label, 'conf': conf, 'decoded': False})

    def markDecoded(self, label, conf):
        for d in reversed(self.detections):  # reversed so we match the most recent detection first
            if d['label'] == label and abs(d['conf'] - conf) < 0.01 and not d['decoded']:  # fuzzy conf match since floats aren't exact
                d['decoded'] = True
                break

    def thresholdSweep(self, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.3, 0.85, 0.05)
        print("\nConf threshold sweep (decode rate):")
        for t in thresholds:
            above = [d for d in self.detections if d['conf'] >= t]  # all detections at or above this threshold
            if not above:
                print(f"  {t:.2f} → no detections")
                continue
            rate = sum(d['decoded'] for d in above) / len(above) * 100  # what fraction were successfully decoded
            print(f"  {t:.2f} → {rate:.1f}%  ({len(above)} detections)")

    def report(self):
        print("\n--- Telemetry Report ---")

        # Latency
        lats = np.array(self.latenciesMs)
        print(f"\nInference latency over {len(lats)} frames:")
        print(f"  p50 : {np.percentile(lats, 50):.1f} ms")
        print(f"  p95 : {np.percentile(lats, 95):.1f} ms")
        print(f"  mean: {lats.mean():.1f} ms")

        # Decode rate by class
        for cls in ['barcode', 'qr']:
            subset = [d for d in self.detections if d['label'] == cls]
            if not subset:
                continue
            rate = sum(d['decoded'] for d in subset) / len(subset) * 100
            print(f"\n{cls}: {rate:.1f}% decode rate ({len(subset)} detections)")

        self.thresholdSweep()

    
