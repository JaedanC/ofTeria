#pragma once
#ifndef OFX_GAME_ENGINE_H
#define OFX_GAME_ENGINE_H

#include "ofMain.h"

class ofxGameState;
class ofxGameEngine
{
public:

	inline int StackSize() const {
		return states.size();
	}

	inline ofxGameState* getState() const {
		return states.back();
	}

	void ChangeState(ofxGameState* state);
	void PushState(ofxGameState* state);
	void PopState();

	void setup();
	void update();
	void draw();

	inline static ofxGameEngine* Instance() {
		return &instance;
	}

protected:
	ofxGameEngine() {}

private:
	static ofxGameEngine instance;
	vector<ofxGameState*> states;
};
#endif /* OFX_GAME_ENGINE_H */