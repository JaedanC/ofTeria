#pragma once
#include "ofMain.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "../ConsoleState.h"
void bind(vector<string> parameters) {
	const string& str = parameters[0];
	ofxGameEngine::Instance()->getKeyboardInput()->registerAlias(parameters[1], str);
}

void clear(vector<string> parameters) {
	ConsoleState::Instance()->clearHistory();
}