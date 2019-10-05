#pragma once
#ifndef CONSOLE_STATE_H
#define CONSOLE_STATE_H
#include "ofMain.h"
#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"

class ConsoleState : public ofxGameState
{
private:
	static ConsoleState instance;
	ofVec2f screenPos;
	int width, height;

	string currentCommand;
	deque<string> history;

protected:
	ConsoleState()
		: ofxGameState(
			true, // updateTransparent
			true, // drawTransparent
			"ConsoleState"
		) {}
public:
	void setup();
	void update(ofxGameEngine* game);
	void draw(ofxGameEngine* game);

	void clearHistory();
	void cullHistory(unsigned int limit);

	static ConsoleState* Instance() {
		return &instance;
	}
};
#endif /* CONSOLE_STATE_H */