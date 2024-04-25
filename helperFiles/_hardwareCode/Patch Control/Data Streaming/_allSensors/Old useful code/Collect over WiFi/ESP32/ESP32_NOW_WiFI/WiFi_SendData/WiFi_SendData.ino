// ********************************** Import Libraries ********************************** //

// WiFi Libraries
#include <esp_wifi.h>
#include <esp_now.h>
#include <WiFi.h>
#include <SPI.h>
// ESP32 ADC Library
#include "esp_adc_cal.h"
// PPG Libraries
#include <SparkFun_Bio_Sensor_Hub_Library.h>

// **************************** Initialize General Variables **************************** //

// Time Variables
const unsigned long oneSecMicro = pow(10,6);
unsigned long beginSamplingTime;
unsigned long endSamplingTime;
unsigned long previousMicros;
unsigned long currentMicros;
unsigned long currentSecond;
unsigned long lastMicros;
int currentMinute;
int currentHour;
// String-Form Time Variables
String currentSecond_String;
String currentMicros_String;
// Keep track of loop time
unsigned long totalLoopTime;
unsigned long startLoopTime;

// Analog Pins
const byte allAdcChannels[] = {A0, A4, A5, A6};
// ESP32 Pins
adc1_channel_t channelEOG = ADC1_CHANNEL_0;
adc1_channel_t channelEEG = ADC1_CHANNEL_4;
adc1_channel_t channelGSR = ADC1_CHANNEL_5;
adc1_channel_t channelTemp = ADC1_CHANNEL_6;
// Specify ADC Parameters
int numberOfReadsADC = 30;
float calibratedVolts;

// Streaming Variables
int readingEOG;
int readingEEG;
int readingGSR;
int readingTemp;
// String-Form Variables
String eogVertical_String;
String eeg_String;
String gsr_String;
String temperature_String;

// ****************************** Initialize WiFi Variables ***************************** //

// Buffer for Serial Printing
const int maxLengthSending = 20;
char sendingMessage[maxLengthSending];

// Broadcasting Variables
uint8_t broadcastAddress[] = {0x7C, 0xDF, 0xA1, 0xF3, 0xCB, 0xBC};
esp_now_peer_info_t peerInfo;
// Assuming No Data MisHandled
esp_err_t recieverReply = ESP_OK;

// ************************************************************************************** //
// ********************************** Helper Functions ********************************** //

byte compressBytes(uint8_t leftInt, uint8_t rightInt) {
    // Convert to Bytes
    byte leftByte = byte(leftInt);
    byte rightByte = byte(rightInt);
    // Compress to One Byte
    byte compressedByte = leftByte << 4 | (0x0F & rightByte);

    return compressedByte;
}

int calibrateADC(float adcRead) {
    // Convert to Volts: THIS IS SPECIFIC TO THIS ESP32!!!
    // Edge Effect: Value too low
    if (adcRead < 500) {
        calibratedVolts = -3.07543411e-13*pow(adcRead, 4) + 4.61714400e-10*pow(adcRead, 3) + -2.34442411e-07*pow(adcRead, 2) + 8.66338357e-04*adcRead + 2.86563719e-02;
    } else {
        calibratedVolts = 1.74592785e-21*pow(adcRead, 6) + -2.36105943e-17*pow(adcRead, 5) + 1.16407137e-13*pow(adcRead, 4) + -2.64411520e-10*pow(adcRead, 3) + 2.74206734e-07*pow(adcRead, 2) + 6.95916329e-04*adcRead + 5.09256786e-02;
    }
    
    // Convert Back to 12-Bits
    return round(calibratedVolts*(4096/3.3));
}

String padZeros(unsigned long number, int totalLength) {
    String finalNumber = String(number);
    int numZeros = totalLength - finalNumber.length();
    for (int i = 0; i < numZeros; i++) {
      finalNumber = "0" + finalNumber;
    }
    return finalNumber;
}

void printBytes(byte inputByte) {
    for (int i = 7; i >= 0; i--) {
        bool bitVal = bitRead(inputByte, i);
        Serial.print(bitVal);
    }
    Serial.println();
}

// ************************************************************************************** //
// *********************************** Setup Functions ********************************** //

void setupADC() {
    // Attach ADC Pins
    for (int adcChannelInd = 0; adcChannelInd < (int) sizeof(allAdcChannels)/sizeof(allAdcChannels[0]); adcChannelInd++) {
        adcAttachPin(allAdcChannels[adcChannelInd]);
    }
    // Configure ADC Pins
    adc1_config_channel_atten(channelEOG, ADC_ATTEN_DB_11); 
    adc1_config_channel_atten(channelEEG, ADC_ATTEN_DB_11); 
    adc1_config_channel_atten(channelGSR, ADC_ATTEN_DB_11); 
    adc1_config_channel_atten(channelTemp, ADC_ATTEN_DB_11);  

    // Set ADC Calibration
    adc_set_clk_div(2);
    analogReadResolution(12);  // Initialize ADC Resolution (Arduino Nano 33 IoT Max = 12)
    adc1_config_width(ADC_WIDTH_12Bit);
    adc_set_data_width(ADC_UNIT_1, ADC_WIDTH_BIT_12);
    // Calibrate ADC  
    esp_adc_cal_characteristics_t adc_chars;
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11, ADC_WIDTH_BIT_12, 1000, &adc_chars);
}

void connectToPeer() {
    // Establish a WiFi Station
    WiFi.mode(WIFI_STA);
    esp_wifi_start();
    // Print the MAC Address of the Device
    Serial.print("MAC Address:");
    Serial.println(WiFi.macAddress());

    // Init ESP-NOW
    if (esp_now_init() != ESP_OK) {
      Serial.println("Error initializing ESP-NOW");
      return;
    }
    
    // Register peer
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = 0;  
    peerInfo.encrypt = false;
    
    // Add peer        
    if (esp_now_add_peer(&peerInfo) != ESP_OK){
      Serial.println("Failed to add peer");
      return;
    }

    // ESP-Now General Setup
    esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_1M_L);  // Set Data Transfer Rate
    esp_wifi_set_storage(WIFI_STORAGE_FLASH);  // Store Data in Flash and Memory
    esp_event_loop_create_default();
    esp_netif_init();
}

// ************************************************************************************** //
// *********************************** Arduino Setup ************************************ //

// Setup Arduino; Runs Once
void setup() {
    //Initialize serial and wait for port to open:
    Serial.begin(115200);     // Use 115200 baud rate for serial communication
    Serial.flush();           // Flush anything left in the Serial port
    Wire.begin();
    
    // Setup ESP32
    setupADC(); // ADC Calibration
    connectToPeer();  // Initialize WiFi

    // Start the Timer at Zero
    currentSecond = 0;
    previousMicros = micros();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

void loop() {

    startLoopTime = micros();
    beginSamplingTime = micros() - previousMicros;
    // Reset Variables
    readingEOG = 0; readingEEG = 0;
    readingGSR = 0; readingTemp = 0;
    // Multisampling Analog Read
    for (int i = 0; i < numberOfReadsADC; i++) {
        // Stream in the Data from the Board
        readingEOG += adc1_get_raw(channelEOG);
        readingEEG += adc1_get_raw(channelEEG);
        readingGSR += adc1_get_raw(channelGSR);
        readingTemp += adc1_get_raw(channelTemp);
    }
    readingEOG = calibrateADC(readingEOG/numberOfReadsADC);
    readingEEG = calibrateADC(readingEEG/numberOfReadsADC);
    readingGSR = calibrateADC(readingGSR/numberOfReadsADC);
    readingTemp = calibrateADC(readingTemp/numberOfReadsADC);
    // Record Final Time
    endSamplingTime = micros() - previousMicros;

    // Record the Time the Signals Were Collected (from Previous Point)
    currentMicros = (beginSamplingTime + endSamplingTime)/2;
    while (currentMicros >= oneSecMicro) {
        currentSecond += 1;
        currentMicros -= oneSecMicro;
    }
    
    // Convert Data into String
    eogVertical_String = padZeros(readingEOG, 4);
    eeg_String = padZeros(readingEEG, 4);
    gsr_String = padZeros(readingGSR, 4);
    temperature_String = padZeros(readingTemp, 4);
    // Convert Times into String
    currentSecond_String = padZeros(currentSecond, 2);
    currentMicros_String = padZeros(currentMicros, 6);

    //Serial.println(gsr_String);
    //Serial.flush();
    // Serial.println(currentSecond_String + "." + currentMicros_String);

    // Compile Sensor Data to Send
    sprintf(sendingMessage, "%c%c%c%c%c%c%c%c%c%c%c%c", 
        compressBytes(currentSecond_String[0], currentSecond_String[1]),
        compressBytes(currentMicros_String[0], currentMicros_String[1]), compressBytes(currentMicros_String[2], currentMicros_String[3]), compressBytes(currentMicros_String[4], currentMicros_String[5]),
        compressBytes(eogVertical_String[0], eogVertical_String[1]), compressBytes(eogVertical_String[2], eogVertical_String[3]),
        compressBytes(eeg_String[0], eeg_String[1]), compressBytes(eeg_String[2], eeg_String[3]),
        compressBytes(gsr_String[0], gsr_String[1]), compressBytes(gsr_String[2], gsr_String[3]),
        compressBytes(temperature_String[0], temperature_String[1]), compressBytes(temperature_String[2], temperature_String[3])
    );
    // Send Sensor Data Using ESP-NOW
    // recieverReply = esp_now_send(broadcastAddress, (uint8_t *) &sendingMessage, sizeof(sendingMessage));
    Serial.write(sendingMessage, 13);
    Serial.println();

    // If Data Sent
    if (recieverReply == ESP_OK) {
        // Keep Track of Time Gap Between Points
        previousMicros = previousMicros + currentMicros + oneSecMicro*currentSecond;
    }

    // Reset Parameters
    currentSecond = 0;
    memset(&sendingMessage[0], 0, sizeof(sendingMessage));
    
    // Add Delay for WiFi to Send Data
    totalLoopTime = micros() - startLoopTime;
    if (3000 - totalLoopTime > 100) {delayMicroseconds(3000 - totalLoopTime);}
}
