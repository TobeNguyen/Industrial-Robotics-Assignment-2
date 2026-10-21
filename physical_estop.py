# Including libraries
import threading
import time

import gui

try:
    import serial
    import serial.tools.list_ports
except ImportError:                     # pyserial not installed: the simulation still works
    serial = None

# Serial baudrate (must match huart2 in main.c)
BAUD = 115200

# ST-LINK Virtual COM Port identifiers
STLINK_VID = 0x0483

_thread = None

# E-STOP state the Nucleo is known to have (None = unknown, forces a send after every (re)connect)
_board_estop = None

# Return the Serial port of the Nucleo, or None if no board is plugged in
def find_nucleo_port():
    ports = list(serial.tools.list_ports.comports())

    # 1) Prefer a match on the ST vendor ID (works on macOS, Windows, Linux)
    for p in ports:
        if p.vid == STLINK_VID:
            return p.device

    # 2) Fallback: match by description text
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if "stlink" in text or "st-link" in text or "stmicroelectronics" in text:
            return p.device

    # 3) Last resort on macOS: any usbmodem device
    for p in ports:
        if "usbmodem" in p.device:
            return p.device

    return None

# Process one received line: same effect as the E-STOP and RESET PROCESS buttons of the GUI
def handle_line(line):
    global _board_estop
    if line == "ESTOP":
        gui.log_step("Physical E-STOP pressed")
        gui.request_stop()
        _board_estop = True             # the board is already in E-STOP, nothing to send back
    elif line == "RESET":
        # ignored if the GUI already cleared the E-STOP
        if gui._estopped():
            gui.log_step("Physical E-STOP released")
            gui.reset_process()
        _board_estop = gui._estopped()
    elif line:
        print(f"[physical e-stop] unknown message: {line}")

# Sends the E-STOP state to the Nucleo when it differs from what the board knows
# (GUI E-STOP, collision / singularity stop, RESET PROCESS)
def sync_to_board(ser):
    global _board_estop
    now = gui._estopped()
    if now != _board_estop:
        ser.write(b"ESTOP\n" if now else b"RESET\n")
        _board_estop = now

# Background loop: waits for the board, reads lines, reconnects if the cable is unplugged
def _listen():
    global _board_estop
    announced_waiting = False
    while True:
        port = find_nucleo_port()
        if port is None:
            if not announced_waiting:
                print("[physical e-stop] no Nucleo found, waiting (GUI E-STOP still works)")
                announced_waiting = True
            time.sleep(1)
            continue
        announced_waiting = False
        try:
            with serial.Serial(port, BAUD, timeout=0.1) as ser:
                gui.log_step(f"Physical E-STOP connected on {port}")
                _board_estop = None     # send the current state once after (re)connect
                while True:
                    handle_line(ser.readline().decode(errors="ignore").strip())
                    sync_to_board(ser)
        except (serial.SerialException, OSError):
            gui.log_step("Physical E-STOP disconnected, reconnecting...")
            time.sleep(1)

# Starts the listener in a daemon thread (it ends together with the program)
def start():
    global _thread
    if serial is None:
        print("[physical e-stop] pyserial is not installed (pip install pyserial), skipped")
        return False
    if _thread is not None and _thread.is_alive():
        return True
    _thread = threading.Thread(target=_listen, daemon=True)
    _thread.start()
    return True