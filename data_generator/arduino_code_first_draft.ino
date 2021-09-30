#include <SPI.h>
#include <Controllino.h>
#include <ArduinoJson.h>
#include <Ethernet.h>

EthernetClient client;
String http_request;
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED };//MAC of controllino
// the dns server ip
//IPAddress dnServer(192, 168, 0, 1);
// the router's gateway address:
//IPAddress gateway(192, 168, 0, 1);
// the subnet:
//IPAddress subnet(255, 255, 255, 0);
//the IP address is dependent on your network
IPAddress ip(192, 168, 2, 2);
//byte ip[] = { 10, 131, 13, 58};//ip of controllino
byte REST_SERVER_IP[] = { 192, 168, 2, 1};//actual ip of cranberry-apm
int REST_SERVER_PORT = 5000;
String REST_ENDPOINT = "/hardware/current";
int counter = 0;
int fan_1_temp = 0;
int fan_2_temp = 0;
int fan_3_temp = 0;
int pump_temp = 0;
int motor_1_temp = 0;
int motor_2_temp = 0;

void setup() {
  Serial.begin(9600);
  //Ethernet.begin(mac, ip, dnServer, gateway, subnet);
  Ethernet.begin(mac, ip);
  Serial.println("Start...");
  
  http_request = build_http_request();
  
  //Digital Outputs PWM
  pinMode(CONTROLLINO_D0, OUTPUT);
  pinMode(CONTROLLINO_D1, OUTPUT);
  pinMode(CONTROLLINO_D2, OUTPUT);
  pinMode(CONTROLLINO_D3, OUTPUT);
  pinMode(CONTROLLINO_D4, OUTPUT);
  pinMode(CONTROLLINO_D5, OUTPUT);
}

void loop() {
  Serial.print("loop: ");
  Serial.println(counter);
  DynamicJsonDocument doc = send_request();
  set_output(doc);
  counter = counter + 1;
}

bool connect_client(){
  int connection = 0;
  int timeout = 0;
  while(connection <= 0 || timeout > 3){
    connection = client.connect(REST_SERVER_IP, REST_SERVER_PORT);
    if(connection < 0){
      Serial.println("connection failed");
      Serial.print("return code:");
      Serial.println(connection);
      return false;
    }
    else{
      Serial.println("connection successful");
      return true;
    }
  }
}

DynamicJsonDocument send_request() {
  int timeout_availability = 0;
  // Send HTTP Request to REST API
  DynamicJsonDocument doc(1024);
  while(!client.connected()){
    Serial.println("disconnected");
    if(connect_client()){
      byte bytes_written = client.print(http_request);
      if(bytes_written == 0){
        return emergency_shutdown();
      }
    }
    else{
      return emergency_shutdown();
    }
  }

  while(!client.available()){
    Serial.println("is not available");
    if(timeout_availability > 4){
      return emergency_shutdown();
    }
    timeout_availability = timeout_availability + 1;
  }
  
  while(client.available()) {
    String response_line = client.readStringUntil('\n');
    Serial.println(response_line);
    if(response_line.indexOf("fan_1") > 0){
      deserializeJson(doc, response_line);
      Serial.println(response_line);
      client.stop();
      return doc;
    }
  }
}

String build_http_request() {
  String httprequest = "GET ";
  httprequest += REST_ENDPOINT;
  httprequest += " HTTP/1.1\r\n";
  httprequest += "Connection: close\r\n\r\n";
  return httprequest;
}

void set_output(DynamicJsonDocument doc) {
  int fan_1 = percent_to_analog(doc["fan_1"]);
  int fan_2 = percent_to_analog(doc["fan_2"]);
  int fan_3 = percent_to_analog(doc["fan_3"]);
  int pump = percent_to_analog(doc["pump"]);
  int motor_1 = percent_to_analog(doc["motor_1"]);
  int motor_2 = percent_to_analog(doc["motor_2"]);

  if(fan_1 != fan_1_temp || fan_2 != fan_2_temp || fan_3 != fan_3_temp || pump != pump_temp || motor_1 != motor_1_temp || motor_2 != motor_2_temp){
    if(fan_1_temp == 0 && fan_1 > 0){
      analogWrite(CONTROLLINO_D0, 255);
    }
    if(fan_2_temp == 0 && fan_2 > 0){
      analogWrite(CONTROLLINO_D1, 255);
    }
    if(fan_3_temp == 0 && fan_3 > 0){
      analogWrite(CONTROLLINO_D2, 255);
    }
    if(pump_temp == 0 && pump > 0){
      analogWrite(CONTROLLINO_D3, 255);
    }
    fan_1_temp = fan_1;
	  fan_2_temp = fan_2;
	  fan_3_temp = fan_3;
	  pump_temp = pump;
	  motor_1_temp = motor_1;
	  motor_2_temp = motor_2;
	  delay(3000);
  }
  
  Serial.print(fan_1);
  Serial.print("\t");
  Serial.print(fan_2);
  Serial.print("\t");
  Serial.print(fan_3);
  Serial.print("\t");
  Serial.print(pump);
  Serial.print("\t");
  Serial.print(motor_1);
  Serial.print("\t");
  Serial.println(motor_2);
  
  analogWrite(CONTROLLINO_D0, fan_1);
  analogWrite(CONTROLLINO_D1, fan_2);
  analogWrite(CONTROLLINO_D2, fan_3);
  analogWrite(CONTROLLINO_D3, pump);
  analogWrite(CONTROLLINO_D4, motor_1);
  analogWrite(CONTROLLINO_D5, motor_2);
}

int percent_to_analog(float value){
  int map_value = value;
  return map(map_value, 0, 100, 0, 255);
}

DynamicJsonDocument emergency_shutdown(){
  String all_zero = "{\"fan_1\":0.0,\"fan_2\":0.0,\"fan_3\":0.0,\"motor_1\":0.0,\"motor_2\":0.0,\"pump\":0.0}";
  DynamicJsonDocument doc(1024);
  deserializeJson(doc, all_zero);
  return doc;
}