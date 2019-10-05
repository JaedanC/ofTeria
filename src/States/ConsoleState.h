#pragma once
#ifndef CONSOLE_STATE_H
#define CONSOLE_STATE_H
#include "ofMain.h"
#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "../Keyboard/KeyboardInput.h"

class ConsoleState : public ofxGameState, public KeyboardCallbacks
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
	void keyPressed(int key);
	void keyReleased(int key);

	void setup();
	void update(ofxGameEngine* game);
	void draw(ofxGameEngine* game);

	void submitCommand(string& command);
	void clearHistory();
	void cullHistory(unsigned int limit);

	int lastHistoryMarker = 0;
	int maxHistorySize = 11;

	static ConsoleState* Instance() {
		return &instance;
	}
};
#endif /* CONSOLE_STATE_H */