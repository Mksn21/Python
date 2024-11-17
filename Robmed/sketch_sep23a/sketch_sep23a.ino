// Define PWM arrays
int Pwm_FIS_kiri[3][3] = {{10, 10, 10}, {10, 80, 50}, {10, 50, 80}};
int Pwm_FIS_kanan[3][3] = {{80, 50, 10}, {50, 80, 10}, {10, 10, 10}};

float uerror[3] = {0, 0, 0};
float uderror[3] = {0, 0, 0};
float FisPWN[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
float upwm[3] = {10, 50, 80};

float error = 0;
float derror = 0;

// Function to calculate fuzzy error
void fuzzerror() {
    if (error < 0) {
        uerror[0] = 1 - (error + 6) / 6.0;
    } else {
        uerror[0] = 0;
    }
    
    if (error > -6 && error <= 0) {
        uerror[1] = (error + 6) / 6.0;
    } else if (error > 0 && error <= 6) {
        uerror[1] = 1 - (error) / 6.0;
    } else {
        uerror[1] = 0;
    }
    
    if (error > 0) {
        uerror[2] = error / 6.0;
    } else {
        uerror[2] = 0;
    }
}

// Function to calculate fuzzy derror
void fuzzderror() {
    if (derror < 0) {
        uderror[0] = 1 - (derror + 6) / 6.0;
    } else {
        uderror[0] = 0;
    }
    
    if (derror > -6 && derror <= 0) {
        uderror[1] = (derror + 6) / 6.0;
    } else if (derror > 0 && derror <= 6) {
        uderror[1] = 1 - (derror) / 6.0;
    } else {
        uderror[1] = 0;
    }
    
    if (derror > 0) {
        uderror[2] = derror / 6.0;
    } else {
        uderror[2] = 0;
    }
}

// Function to calculate FisPWN matrix
void fuzzyfikasi() {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            FisPWN[i][j] = min(uderror[i], uerror[j]);
        }
    }
}

void setup() {
    Serial.begin(9600);
    Serial.println("Enter values for error and derror separated by a comma (e.g., 1,-2):");
}

void loop() {
    // Check if data is available on Serial
    if (Serial.available() > 0) {
        // Read input from Serial as a string
        String input = Serial.readStringUntil('\n');
        
        // Split the input string by comma to get error and derror
        int commaIndex = input.indexOf(',');
        if (commaIndex > 0) {
            error = input.substring(0, commaIndex).toFloat();
            derror = input.substring(commaIndex + 1).toFloat();

            // Fuzzify error and derror
            fuzzerror();
            fuzzderror();
            fuzzyfikasi();

            // Defuzzification
            float JumlahFIS = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    JumlahFIS += FisPWN[j][i];
                }
            }
            
            float num_kanan = 0;
            float num_kiri = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    num_kanan += FisPWN[i][j] * Pwm_FIS_kanan[i][j];
                    num_kiri += FisPWN[i][j] * Pwm_FIS_kiri[i][j];
                }
            }
            
            float Pwm_kanan = num_kanan / JumlahFIS;
            float Pwm_kiri = num_kiri / JumlahFIS;

            // Print results
            Serial.print("Pwm_kanan: ");
            Serial.println(Pwm_kanan);
            Serial.print("Pwm_kiri: ");
            Serial.println(Pwm_kiri);
            Serial.println("Enter values for error and derror separated by a comma:");
        }
    }
}
