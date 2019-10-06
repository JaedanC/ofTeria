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
	int lastHistoryMarker = 0;
	int maxHistorySize = 11;

protected:
	ConsoleState();
public:
	virtual void keyPressed(int key);
	virtual void keyReleased(int key);
	virtual void setup();
	virtual void update(ofxGameEngine* game);
	virtual void draw(ofxGameEngine* game);
	virtual void exit();

	/* Adds the current command to the deque and updates the length
	of the history if required.*/
	void submitCommand(string& command);
	void clearHistory();
	/* Limits the size of the history deque to length <limit>. Culls
	the oldest first (cause it's a queue duh). */
	void cullHistory(unsigned int limit);

	static ConsoleState* Instance() {
		return &instance;
	}
};
#endif /* CONSOLE_STATE_H */