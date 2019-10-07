#pragma once
#ifndef OFX_DEBUGGER_H
#define OFX_DEBUGGER_H

#include "ofMain.h"

#define debugPush(x) ofxDebugger::Instance()->push(x)
#define debugDraw ofxDebugger::Instance()->draw

/* Easiest way to show a debug variable is by using the debugPush(string) macro
which is defined in the same header. */
class ofxDebugger {
public:
	vector<string> debugStrings;

	// Add an string to the stack.
	void push(const string& message);

	// Draws all the debugStrings to the screen neatly
	void draw();

	static ofxDebugger* Instance() {
		static ofxDebugger instance;
		return &instance;
	}

protected:
	ofxDebugger() {}
};
#endif /* OFX_DEBUGGER_H */