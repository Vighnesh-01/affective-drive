/*
  Affective-Drive Actuator Controller
  Baud Rate: 9600
  Commands:
    'H' -> HIGH Risk: Red LED ON, Green LED OFF, Pulse Buzzer Alert
    'L' -> LOW Risk:  Green LED ON, Red LED OFF, Silent
*/
#include <Arduino.h>

const int GREEN_LED = 8;
const int RED_LED = 9;
const int BUZZER = 10;

// System State
char currentSignal = 'L';
unsigned long lastBuzzerTime = 0;
bool buzzerState = false;

void setup() {
  // Initialize Serial Communication
  Serial.begin(9600);

  // Initialize Pin Modes
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  // Initial State: System Ready (Green Light)
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED, LOW);
  noTone(BUZZER);
}

void loop() {
  // Check for incoming serial data from Python
  if (Serial.available() > 0) {
    char incomingChar = Serial.read();

    // Only update state if valid command is received
    if (incomingChar == 'H' || incomingChar == 'L') {
      currentSignal = incomingChar;
    }
  }

  // Actuation Logic
  if (currentSignal == 'H') {
    // High-Risk Alert Mode
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(RED_LED, HIGH);

    // Non-blocking buzzer alert tone (1000 Hz pulse every 200ms)
    unsigned long currentTime = millis();
    if (currentTime - lastBuzzerTime >= 200) {
      lastBuzzerTime = currentTime;
      buzzerState = !buzzerState;

      if (buzzerState) {
        tone(BUZZER, 1000); // 1 kHz Alert Frequency
      } else {
        noTone(BUZZER);
      }
    }
  } else {
    // Normal / Low-Risk Mode
    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED, LOW);
    noTone(BUZZER);
  }
}