#include <Stepper.h>
#include <Servo.h>

const int stepsPerRevolution = 200; 
Stepper myStepper(500, 3,4,5,6);
Servo servo;

String incomingString= "";

void setup() {
  myStepper.setSpeed(24);
  servo.attach(8);
  Serial.begin(9600);
}

void loop() {
  
  Serial.println("Prime skittle");
  myStepper.step(400); 
//  delay(3000);

  String colors[] = {"red", "orange","yellow","green","violet"};

  int x=1;
//  Serial.println("Enter a skittle that was primed : \n 0 - Red \n 1 - Orange \n 2 - Yellow \n 3 - Green \n 4 - Violet");
//  int x = Serial.read();
//  delay(5000);

  if(colors[x].equals("red")){
    servo.write(75);
  }
  else if(colors[x].equals("orange")){
    servo.write(95);
  }
  else if(colors[x].equals("yellow")){
    servo.write(120);
  }
  else if(colors[x].equals("green")){
    servo.write(140);
  }
  else if(colors[x].equals("violet")){
    servo.write(160);
  } 

  x = x + 1 ;

  Serial.println("Rotating back to original spot...");
  myStepper.step(1680);
//  delay(3000);
 }
