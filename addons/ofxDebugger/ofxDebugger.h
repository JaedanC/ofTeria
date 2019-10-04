#pragma once
#ifndef OFX_DEBUGGER_H
#define OFX_DEBUGGER_H

#include "ofMain.h"

class ofxDebugger {
public:
	vector<string> debugStrings;

	void push(const string& message);
	void draw();
};
#endif /* OFX_DEBUGGER_H */