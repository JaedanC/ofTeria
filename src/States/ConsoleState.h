#pragma once
#ifndef CONSOLE_STATE_H
#define CONSOLE_STATE_H
#include "ofMain.h"
#include "ofxGameStates/ofxGameState.h"
#include "ofxGameStates/ofxGameEngine.h"
#include "../Keyboard/KeyboardInput.h"
#include "Console/ConsoleParser.h"

struct ConsoleEntry {
	ofColor colour;
	vector<string> message;

	ConsoleEntry(vector<string> message, ofColor colour = ofColor::white)
		: message(message), colour(colour)
	{}
};

class ConsoleState : public ofxGameState, public KeyboardCallbacks
{
private:
	ofVec2f screenPos;
	int width = 330;
	int height = 250;
	int screenGap = 15;

	string currentCommand;
	deque<ConsoleEntry> consoleHistory;
	deque<string> commandHistory;

	int commandHistoryMarker = 0;
	int commandHistoryMaxSize = 5;
	int showHistoryLines = 15;
	int consoleHistoryMarker = 0;
	int consoleHistoryMaxSize = 20;
	ofColor colorCommand = ofColor(255, 255, 50);
	ofColor colorPass = ofColor(50, 255, 50);

	ConsoleParser consoleParser;

	static ConsoleState instance;

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
	void addText(ConsoleEntry& entry);
	void addText(vector<string>& entries, ofColor colour = ofColor(255, 255, 255));
	void addText(string& entry, ofColor colour = ofColor(255, 255, 255));

	void clearHistory();

	/* Limits the size of the history deque to length <limit>. Culls
	the oldest first (cause it's a queue duh). */
	void cullHistory();

	static ConsoleState* Instance() {
		return &instance;
	}
};


#endif /* CONSOLE_STATE_H */