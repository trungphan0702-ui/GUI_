import sounddevice as sd
from typing import List, Tuple


def list_devices() -> Tuple[List[str], List[str]]:
    inputs, outputs = [], []
    try:
        for i, dev in enumerate(sd.query_devices()):
            name = f"{i}: {dev['name']}"
            if dev.get('max_input_channels', 0) > 0:
                inputs.append(name)
            if dev.get('max_output_channels', 0) > 0:
                outputs.append(name)
    except Exception:
        pass
    return inputs, outputs


def parse_device(selection: str):
    try:
        return int(selection.split(":", 1)[0])
    except Exception:
        return None


def default_samplerate(out_dev):
    try:
        dev_info = sd.query_devices(out_dev, 'output')
        return int(dev_info['default_samplerate'])
    except Exception:
        try:
            dev_info = sd.query_devices(None, 'output')
            return int(dev_info['default_samplerate'])
        except Exception:
            return 48000
