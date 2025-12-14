# Audio Analysis Toolkit

## Run locally (Windows)

1. **Create & activate a virtual environment**
   ```powershell
   py -3 -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
3. **Run offline self-test suite** (no soundcard needed)
   ```powershell
   python -m tests.self_test --mode offline
   ```
4. **Run hardware smoke test** (requires working input/output devices and routing)
   ```powershell
   python -m tests.self_test --mode hardware --in-dev <input_device_id> --out-dev <output_device_id>
   ```

Tips:
- List available devices and environment details with:
  ```powershell
  python tools/print_env.py
  ```
- On Windows, ensure microphone/speaker permissions are enabled for the terminal session.
