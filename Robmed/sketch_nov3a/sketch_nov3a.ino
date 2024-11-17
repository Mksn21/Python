#include <Servo.h>

Servo myservo;  // create servo object to control a servo
int pos = 0; 

void setup() {
  myservo.attach(9);
}

void loop() {
 myservo.write(90);              // tell servo to go to position in variable 'pos'
    delay(15);
}
