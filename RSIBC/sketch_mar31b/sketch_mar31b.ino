void setup() {
  Serial.begin(115200);
}

void loop() {
  for (int i = 0; i < 360; i += 5) {
    float sine_1 = 1 * sin(i * M_PI / 180);
    float sine_2 = 2 * sin((i + 90) * M_PI / 180);
    float sine_3 = 5 * sin((i + 180) * M_PI / 180);
    float sine_4 = 3 * sin((i + 270) * M_PI / 180);

    Serial.print(sine_1);
    Serial.print("\t"); // a tab '\t' or space ' ' character is printed between the two values.
    Serial.print(sine_2);
    Serial.print("\t"); // a tab '\t' or space ' ' character is printed between the two values.
    Serial.println(sine_3); // the last value is terminated by a carriage return and a newline characters.
    Serial.print("\t");
    Serial.println(sine_4);
    Serial.print("\n");
    delay(100);
  }
}
