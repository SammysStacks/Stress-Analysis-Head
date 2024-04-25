// ********************************** Import Libraries ********************************** //
// Fast Analog Read Library
#include "avdweb_AnalogReadFast.h"

// **************************** Initialize General Variables **************************** //

// Label pins
const int heatingPadPin = 9;      // LED connected to digital pin 9

// ADC parameters
float readingADC;
int numberOfReadsADC = 40;

// Heating pad parameters
boolean connectToPython = false;
int voltageReading;
int voltageInput = 255; // Voltages go from 100 - 160??

// Feedback control
int tempReading;
float tempVolts;
float calibratedTemp;
float feedbackTemp;

// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

// Read and average ADC values. Throw out initial reads.
float readADC(int channel) {
    // Throw out the first result
    analogReadFast(channel);
    
    readingADC = 0.00000;
    // Multisampling Analog Read
    for (int i = 0; i < numberOfReadsADC; i++) {
        // Stream in the Data from the Board
        readingADC += analogReadFast(channel);
    }
    // Calibrate the ADC value - BOARD SPECIFIC!
    readingADC = readingADC/numberOfReadsADC;

    return readingADC;
}

float calibrateTemp_TMP36(float tempADC) {
    // Converted ADC to Voltage.
    tempVolts = tempADC * (5.0000/1023.0000);
    //Serial.println(tempVolts);

    // Convert Voltage to temperature.
    calibratedTemp = 100.00*tempVolts - 50.000;

    // Return the 12-Bit value
    return calibratedTemp;
}


// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// the setup routine runs once when you press reset:
void setup() {
    //Initialize serial and wait for port to open.
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    Serial.setTimeout(1);
    Serial.flush();           // Flush anything left in the Serial port

    // Set pins as output pins.
    pinMode(heatingPadPin, OUTPUT);  
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

// the loop routine runs over and over again forever:
void loop() {
    // Read in stress level from python
    if (connectToPython) {
        while (!Serial.available());
        voltageInput = Serial.readString().toInt();
    }
    
    for (int voltageInput = 0; voltageInput < 255; voltageInput += 50) {
      voltageInput = 150;
      // Adjust the temperature on the pad
      analogWrite(heatingPadPin, voltageInput); // analogRead values go from 0 to 1023, analogWrite values from 0 to 255
      //delay(100);                               // Delay in between reads for stability

      // Feedback control of temperature
      tempReading = readADC(A0);
      // Serial.println(tempReading * 5.000 / 1023.000);
      feedbackTemp = calibrateTemp_TMP36(tempReading);
      
      //Serial.println(String(feedbackTemp) + ", 20, 35");

      // Read in the temperature
      //Serial.println(String(voltageReading) + ",1023,0");

      int filterADC = readADC(A1);
      float filterVoltage = filterADC * (5.0000/1023.0000);
      Serial.println(String(voltageInput) + " " + String(filterVoltage) + " 0 5");
    }
}
