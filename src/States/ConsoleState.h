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

class ConsoleCursor {
private:
	int cursor;
	int selectionAnchor;
public:
	ConsoleCursor() : cursor(0), selectionAnchor(0) {}

	inline bool highlighting() { return cursor != selectionAnchor; }
	inline int getCursor() { return cursor;  }
	
	inline void set(int num) {
		cursor = num;
		reset();
	}

	ConsoleCursor& operator++() {
		++cursor;
		return *this;
	}

	ConsoleCursor operator++(int) {
		ConsoleCursor result(*this);
		++(*this);
		return result;
	}

	ConsoleCursor& operator--() {
		--cursor;
		return *this;
	}

	ConsoleCursor operator--(int) {
		ConsoleCursor result(*this);
		--(*this);
		return result;
	}

	inline void highlight(int amount) {
		selectionAnchor += amount;
	}

	inline void reset() {
		selectionAnchor = cursor;
	}

	inline void clamp(int min, int max) {
		cursor = ofClamp(cursor, min, max);
		selectionAnchor = ofClamp(selectionAnchor, min, max);
	}

	inline int dist() {
		return ABS(cursor - selectionAnchor);
	}

	inline int left() {
		return MIN(cursor, selectionAnchor);
	}

	inline int right() {
		return MAX(cursor, selectionAnchor);
	}
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

	ConsoleCursor cursor;
	int commandHistoryMarker = 0;
	int commandHistoryMaxSize = 25;
	int showHistoryLines = 15;
	int consoleHistoryMarker = 0;
	int consoleHistoryMaxSize = 50;
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
	virtual void exit();
	virtual void update(ofxGameEngine* game);
	virtual void draw(ofxGameEngine* game);

	/* Adds the current command to the deque and updates the length
	of the history if required.*/
	void submitCommand(const string& command);
	void addText(const ConsoleEntry& entry);
	void addText(vector<string>& entries, ofColor colour = ofColor(255, 255, 255));
	void addText(const string& entry, ofColor colour = ofColor(255, 255, 255));

	void clearHistory();

	/* Limits the size of the history deque to length <limit>. Culls
	the oldest first (cause it's a queue duh). */
	void cullHistory();

	static ConsoleState* Instance() {
		return &instance;
	}
};


#endif /* CONSOLE_STATE_H */