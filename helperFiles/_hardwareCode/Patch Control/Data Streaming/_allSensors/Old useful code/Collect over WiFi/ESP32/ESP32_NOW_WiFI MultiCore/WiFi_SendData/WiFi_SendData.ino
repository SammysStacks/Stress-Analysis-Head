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

// Thread0 Variables Strings
String currentSecond_String_Thread0 = "1";
String currentMicros_String_Thread0 = "1";
String ppgChannel_String_Thread0_Hold;
String currentMicros_String_Thread0_Hold;
String currentSecond_String_Thread0_Hold;
// Thread0 Variables Times
unsigned long currentMicros_Thread0;
unsigned long currentSecond_Thread0;
unsigned long previousMicros_Thread0;
unsigned long endSamplingTime_Thread0;
unsigned long beginSamplingTime_Thread0;

// Analog Pins
const byte ADC0 = A0;
const byte ADC5 = A4;
// ESP32 Pins
adc1_channel_t channelEOG = ADC1_CHANNEL_0;
adc1_channel_t channelGSR = ADC1_CHANNEL_4;
// Specify ADC Parameters
int numberOfReadsADC = 2000;
float calibratedVolts;

const int SCL_Pin = 47;
const int SDA_Pin = 21;
// Set Sample Width and Sampling Frequency for PPG. Note: There is a Dependance Between Them
int width = 215;    // Possible widths: 69, 118, 215, 411us
int samples = 800; // Possible samples: 50, 100, 200, 400, 800, 1000, 1600, 3200 samples/second
// Set the Reset and MFIO pin for PPG
int resPin = 16;
int mfioPin = 17;
// Takes address, reset pin, and MFIO pin for PPG.
SparkFun_Bio_Sensor_Hub bioHub(resPin, mfioPin); 
bioData body; 

// MultiThreading
TaskHandle_t Task1;
SemaphoreHandle_t Semaphore;
boolean streamDataPPG = true;

// Streaming Variables
int ppgChannel_ADC;
int eogChannelVertical_ADC;
int galvanicSkinResponse_ADC;
// String-Form Variables
String ppgChannel_String = "000000";
String eogChannelVertical_String;
String galvanicSkinResponse_String;

// Buffer for Serial Printing
const int maxLengthSending = 14;
char sendingMessage[maxLengthSending];

// ****************************** Initialize WiFi Variables ***************************** //

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

void setupPPG() {
    bioHub.begin();
    bioHub.configSensorBpm(MODE_ONE);
    bioHub.setPulseWidth(width);
    bioHub.setSampleRate(samples);
    delay(4000);
}

void setupADC() {
    // Attach ADC Pins
    adcAttachPin(ADC0);
    adcAttachPin(ADC5);
    adc1_config_channel_atten(channelGSR, ADC_ATTEN_DB_11); 
    adc1_config_channel_atten(channelEOG, ADC_ATTEN_DB_11);  

     
    // ADC Calibration
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
    
    // Setup ESP32
    setupADC(); // ADC Calibration
    connectToPeer();  // Initialize WiFi
    //connectToClock();

    Wire.begin(SDA_Pin, SCL_Pin);
    // Set up PPG Board/Sensor
    setupPPG();

    // Simple flag, up or down
    Semaphore = xSemaphoreCreateMutex();

    // Setup PPG Threading
    /*xTaskCreatePinnedToCore(
      getPPG, // Function to implement the task
      "Task1", // Name of the task
      10000,  // Stack size in words
      NULL,  // Task input parameter
      0,  // Priority of the task
      &Task1,  // Task handle.
      0); // Core where the task should run
    */
    
    currentSecond = 0;
    currentSecond_Thread0 = 0;
    previousMicros = micros();
    previousMicros_Thread0 = micros();
}

// ************************************************************************************** //
// ************************************ Arduino Loop ************************************ //

void getPPG(void *parameter) {
    // Infinite Loop
    for(;;) {
        // Get the variable in this Scope
        xSemaphoreTake(Semaphore, portMAX_DELAY);
        xSemaphoreGive(Semaphore);

        // Stream Data if Needed
        if (streamDataPPG) {
            beginSamplingTime_Thread0 = micros() - previousMicros_Thread0;
            // Sample PPG
            body = bioHub.readSensorBpm(); // Information from the readSensor function will be saved to our "body" variable.
            ppgChannel_ADC = body.irLed;   // Read in PPG Data
            endSamplingTime_Thread0 = micros() - previousMicros_Thread0;

            // If the reading was good, send it
            if (ppgChannel_ADC != 0) {
                // Record the Time the Signals Were Collected (from Previous Point)
                currentMicros_Thread0 = (beginSamplingTime_Thread0 + endSamplingTime_Thread0)/2;
                while (currentMicros_Thread0 >= oneSecMicro) {
                    currentSecond_Thread0 += 1;
                    currentMicros_Thread0 -= oneSecMicro;
                }
                
                // Convert Data into String
                ppgChannel_String_Thread0_Hold = padZeros(ppgChannel_ADC, 6);
                currentSecond_String_Thread0_Hold = padZeros(currentSecond_Thread0, 2);
                currentMicros_String_Thread0_Hold = padZeros(currentMicros_Thread0, 6).substring(0, 4);
        
                // Flag: Inform the Other Thread
                xSemaphoreTake(Semaphore, portMAX_DELAY);
                streamDataPPG = false;
                xSemaphoreGive(Semaphore);
                
                // Give the PPG a Break
                if (20000 > endSamplingTime_Thread0) {delayMicroseconds(20000 - endSamplingTime_Thread0);}
                //delayMicroseconds(max((long unsigned int)1, 6000 - (micros() - previousMicros_Thread0)));
            } else {
                // The PPG Stopped Working; Could be from Pressure;
                setupPPG();
            }
        }
    }
}

void loop() {

    beginSamplingTime = micros() - previousMicros;
    // Reset Variables
    eogChannelVertical_ADC = 0;
    galvanicSkinResponse_ADC = 0;
    // Multisampling Analog Read
    for (int i = 0; i < numberOfReadsADC; i++) {
        // Stream in the Data from the Board
        eogChannelVertical_ADC += adc1_get_raw(channelEOG);
        galvanicSkinResponse_ADC += adc1_get_raw(channelGSR);
    }
    eogChannelVertical_ADC = calibrateADC(eogChannelVertical_ADC/numberOfReadsADC);
    galvanicSkinResponse_ADC = calibrateADC(galvanicSkinResponse_ADC/numberOfReadsADC);
    // Record Final Time
    endSamplingTime = micros() - previousMicros;

    // Record the Time the Signals Were Collected (from Previous Point)
    currentMicros = (beginSamplingTime + endSamplingTime)/2;
    while (currentMicros >= oneSecMicro) {
        currentSecond += 1;
        currentMicros -= oneSecMicro;
    }
    
    // Convert Data into String
    eogChannelVertical_String = padZeros(eogChannelVertical_ADC, 4);
    galvanicSkinResponse_String = padZeros(galvanicSkinResponse_ADC, 4);
    // Convert Times into String
    currentSecond_String = padZeros(currentSecond, 2);
    currentMicros_String = padZeros(currentMicros, 6);

    // Check if Other Thread is Collecting Data
    if (streamDataPPG) {
        // Send Zeros as a Placeholder as we Wait for the next Datapoint.
        ppgChannel_String = "000000";
    } else {
        xSemaphoreTake(Semaphore, portMAX_DELAY);
        xSemaphoreGive(Semaphore);
        ppgChannel_String = ppgChannel_String_Thread0_Hold;
        currentSecond_String_Thread0 = currentSecond_String_Thread0_Hold;
        currentMicros_String_Thread0 = currentMicros_String_Thread0_Hold;
        //Serial.println(ppgChannel_String);
    }

    Serial.println(galvanicSkinResponse_String);
    //Serial.println(currentSecond_String_Thread0 + "." + currentMicros_String_Thread0);

    // Compile Sensor Data to Send
    sprintf(sendingMessage, "%c%c%c%c%c%c%c%c%c%c%c", 
        compressBytes(currentSecond_String[0], currentSecond_String[1]),
        compressBytes(currentMicros_String[0], currentMicros_String[1]), compressBytes(currentMicros_String[2], currentMicros_String[3]), compressBytes(currentMicros_String[4], currentMicros_String[5]),
        compressBytes(currentSecond_String_Thread0[0], currentSecond_String_Thread0[1]),
        compressBytes(currentMicros_String_Thread0[0], currentMicros_String_Thread0[1]), compressBytes(currentMicros_String_Thread0[2], currentMicros_String_Thread0[3]),
        compressBytes(eogChannelVertical_String[0], eogChannelVertical_String[1]), compressBytes(eogChannelVertical_String[2], eogChannelVertical_String[3]),
        compressBytes(ppgChannel_String[0], ppgChannel_String[1]), compressBytes(ppgChannel_String[2], ppgChannel_String[3]), compressBytes(ppgChannel_String[4], ppgChannel_String[5]),
        compressBytes(galvanicSkinResponse_String[0], galvanicSkinResponse_String[1]), compressBytes(galvanicSkinResponse_String[2], galvanicSkinResponse_String[3])
    );
    // Send Sensor Data Using ESP-NOW
    esp_err_t recieverReply = esp_now_send(broadcastAddress, (uint8_t *) &sendingMessage, sizeof(sendingMessage));
    //recieverReply = ESP_OK;
    
    // If Data Sent
    if (recieverReply == ESP_OK) {
        // Keep Track of Time Gap Between Points
        previousMicros = previousMicros + currentMicros + oneSecMicro*currentSecond;

        // If PPG Data was VENDED from thread0
        if (not streamDataPPG && not ppgChannel_String.equals("000000")) {
            // Track its Time Gap
            previousMicros_Thread0 = previousMicros_Thread0 + currentMicros_Thread0 + oneSecMicro*currentSecond_Thread0;
            // Update Values for the Next Round
            currentSecond_Thread0 = 0;
            streamDataPPG = true;
        }
    }

    // Reset Parameters
    currentSecond = 0;
    memset(&sendingMessage[0], 0, sizeof(sendingMessage));
    
    // Add Delay for WiFi to Send Data
    //Serial.println(endSamplingTime);
    endSamplingTime = micros() - previousMicros;
    if (3000 > endSamplingTime) {delayMicroseconds(3000 - endSamplingTime);}
}
