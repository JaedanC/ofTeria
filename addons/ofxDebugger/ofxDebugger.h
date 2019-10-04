#pragma once
#ifndef OFX_DEBUGGER_H
#define OFX_DEBUGGER_H

#include "ofMain.h"

#define debugPush(x) ofxDebugger::Instance()->push(x)
#define debugDraw ofxDebugger::Instance()->draw

class ofxDebugger {
public:
	vector<string> debugStrings;

	void push(const string& message);
	void draw();

	inline static ofxDebugger* Instance() {
		return &instance;
	}

protected:
	ofxDebugger() {}

private:
	static ofxDebugger instance;
};
#endif /* OFX_DEBUGGER_H */