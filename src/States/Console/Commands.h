#pragma once
#include "ofMain.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "../ConsoleState.h"
void bind(vector<string> parameters) {
	int c = parameters[0][0];
	ofxGameEngine::Instance()->getKeyboardInput()->registerAlias(parameters[1], c);
}

void clear(vector<string> parameters) {
	ConsoleState::Instance()->clearHistory();
}