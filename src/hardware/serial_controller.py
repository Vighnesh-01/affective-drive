import serial
import serial.tools.list_ports
import time

class HardwareController:
    def __init__(self, port=None, baudrate=9600):
        self.mock_mode = False
        self.ser = None

        if port is None:
            ports = list(serial.tools.list_ports.comports())
            if ports:
                port = ports[0].device
            else:
                print("[HARDWARE] No microcontrollers detected. Operating in MOCK MODE.")
                self.mock_mode = True

        if not self.mock_mode:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino reset
                print(f"[HARDWARE] Connected to microcontroller on {port}")
            except Exception as e:
                print(f"[HARDWARE] Serial connection failed ({e}). Reverting to MOCK MODE.")
                self.mock_mode = True

    def send_actuation_signal(self, risk_level):
        """Sends single-byte signals: 'H' for High Risk, 'L' for Low Risk."""
        signal = 'H' if risk_level == "HIGH" else 'L'
        
        if self.mock_mode:
            print(f"[MOCK HARDWARE] Actuator Signal Sent: '{signal}' ({'LED RED / HAPTIC ON' if signal == 'H' else 'LED GREEN'})")
        else:
            try:
                self.ser.write(signal.encode())
            except Exception as e:
                print(f"[HARDWARE ERROR] Failed to send serial data: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()