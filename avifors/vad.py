"""Optional CPU-only Silero ONNX speech detection before GPU admission."""

from __future__ import annotations


def speech_regions(probabilities, duration, threshold=0.5, padding=0.25):
    regions, start, last = [], None, 0
    for index, probability in enumerate(probabilities):
        now = index * 0.032
        if probability >= (threshold if start is None else threshold - 0.15):
            if start is None:
                start = now
            last = min(duration, now + 0.032)
        elif start is not None and now - last >= 0.3:
            if last - start >= 0.096:
                regions.append((max(0, start - padding), min(duration, last + padding)))
            start = None
    if start is not None and last - start >= 0.096:
        regions.append((max(0, start - padding), min(duration, last + padding)))
    merged = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def detect(path, model, threshold=0.5, padding=0.25, stop=None):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])

    def scores():
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        with path.open("rb") as stream:
            while raw := stream.read(1024):
                if stop is not None and stop.is_set():
                    raise InterruptedError("speech detection cancelled")
                waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                waveform = np.pad(waveform, (0, 512 - len(waveform))).reshape(1, 512)
                if not np.any(waveform):
                    # Digital silence cannot contain speech. Reset recurrent
                    # state and skip the network, preserving original timing.
                    state.fill(0)
                    context.fill(0)
                    yield 0.0
                    continue
                value, state = session.run(
                    None,
                    {
                        "input": np.concatenate((context, waveform), axis=1),
                        "state": state,
                        "sr": np.array(16000, dtype=np.int64),
                    },
                )
                context = waveform[:, -64:]
                yield float(value.reshape(-1)[0])

    return speech_regions(scores(), path.stat().st_size / 32000, threshold, padding)


def apply_regions(plan, regions):
    cursor = 0
    for chunk in plan:
        while cursor < len(regions) and regions[cursor][1] <= chunk["start"]:
            cursor += 1
        matching = []
        index = cursor
        while index < len(regions) and regions[index][0] < chunk["end"]:
            matching.append(regions[index])
            index += 1
        if not matching:
            chunk["silent"] = True
        else:
            chunk["infer_start"] = max(chunk["start"], matching[0][0])
            chunk["infer_end"] = min(chunk["end"], matching[-1][1])
    return plan
