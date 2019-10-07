#pragma once
#include "ofMain.h"
#include "../../Keyboard/KeyboardInput.h" 
void bind(vector<string> parameters) {
	int c = parameters[0][0];
	KeyboardInput::Instance()->registerAlias(parameters[1], c);
}