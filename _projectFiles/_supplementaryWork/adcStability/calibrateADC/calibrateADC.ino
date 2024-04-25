// ********************************** Import Libraries ********************************** //

// WiFi Libraries
#include <esp_wifi.h> // ESP32 WiFi Libraries
#include <esp_now.h>  // ESPNOW WiFi communication
#include <WiFi.h>     // Arduino WiFi Libraries
// ESP32 ADC Library
#include "esp_adc_cal.h" // Calibrate ADC for ESP32

// **************************** Initialize General Variables **************************** //

// ADC parameters
int numberOfReadsADC = 100;
// Parameters to store the ADC results
float readingADC;
float sensorValue;
int calibratedValue;
float calibratedVolts;

// Analog Pins
const byte allAdcChannels[] = {A4}; // Connect common (Z) to A3 (analog input)
adc1_channel_t analogChannel = ADC1_CHANNEL_4;


// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

// Read and average ADC values. Throw out initial reads.
float readADC(adc1_channel_t channel) {
    // Throw out the first result
    adc1_get_raw(channel);
    
    readingADC = 0.00000000;
    // Multisampling Analog Read
    for (int i = 0; i < numberOfReadsADC; i++) {
        // Stream in the Data from the Board
        readingADC += adc1_get_raw(channel);
    }

    return readingADC / numberOfReadsADC;
}

// Convert to Volts: THIS IS SPECIFIC TO THIS ESP32!!!
int calibrateADC(float adcRead) {
    // Edge Effect: Value too low
    if (adcRead < 1000) { // 0.0 - 0.8
      float x1_polynomials [] = {1.75280262e-10, -2.95228712e-07,  9.63684730e-04,  5.52278394e-03};
      calibratedVolts = (x1_polynomials[0] * pow(adcRead, 3)) + (x1_polynomials[1] * pow(adcRead, 2)) + (x1_polynomials[2] * adcRead) + x1_polynomials[3];
    }
    else if (adcRead < 3200) { // 0.9 - 2.5
      float x2_polynomials [] = {-6.16258006e-09,  8.07852499e-04,  4.16612572e-02};
      calibratedVolts = ((x2_polynomials[0] * pow(adcRead, 2)) + (x2_polynomials[1] * adcRead) + x2_polynomials[2]);
    }
    else if (adcRead < 4095){ // 2.6 - 3.3
      float x3_polynomials [] = {1.10884628e-09, -1.20158482e-05,  4.39214648e-02, -5.13083813e+01};
      calibratedVolts = ((x3_polynomials[0]  * pow(adcRead, 3)) + (x3_polynomials[1] * pow(adcRead, 2)) + (x3_polynomials[2] * adcRead) + x3_polynomials[3]);
    }
    else {
        calibratedVolts =  3.2;
    }

    // Convert Back to 12-Bits
    return round(calibratedVolts*(4095/3.3));
}

// ************************************************************************************** //
// *********************************** Setup Functions ********************************** //

void setupADC() {
    // Attach ADC Pins
    for (int adcChannelInd = 0; adcChannelInd < (int) sizeof(allAdcChannels)/sizeof(allAdcChannels[0]); adcChannelInd++) {
        adcAttachPin(allAdcChannels[adcChannelInd]);
    }
    // Configure ADC Pins
    adc1_config_channel_atten(analogChannel, ADC_ATTEN_DB_11); 

    // Set ADC Calibration
    adc_set_clk_div(2);
    analogReadResolution(12);  // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    adc1_config_width(ADC_WIDTH_12Bit);
    adc_set_data_width(ADC_UNIT_1, ADC_WIDTH_BIT_12);
    // Calibrate ADC  
    esp_adc_cal_characteristics_t adc_chars;
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1000, &adc_chars);
}

// the setup routine runs once when you press reset:
void setup() {
    //Initialize serial and wait for port to open.
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    Serial.flush();           // Flush anything left in the Serial port

    // Setup ADC
    setupADC();
}

// the loop routine runs over and over again forever:
void loop() {
  
    sensorValue = readADC(analogChannel);
    calibratedValue = calibrateADC(sensorValue);

    // print out the value you read:
    Serial.println(sensorValue);
    
    // print out polyfitted value:
   // Serial.println("Volts: " + String(calibratedValue*3.3/4095));

    delay(1);        // delay in between reads for stability
}
