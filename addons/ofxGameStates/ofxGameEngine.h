#pragma once
#ifndef OFX_GAME_ENGINE_H
#define OFX_GAME_ENGINE_H

#include "ofMain.h"

class KeyboardCallbacks;
class ofxGameState;
class ofxGameEngine
{
private:
	static ofxGameEngine instance;
	vector<ofxGameState*> states;

protected:
	ofxGameEngine() {}

public:
	// Returns the size of the GameState stack.
	int StackSize() const;

	// Returns the GameState on the top of the stack.
	ofxGameState* getState() const;

	// Switch the GameState on the top of the stack to something else.
	void ChangeState(ofxGameState* state);

	// Push a GameState onto the stack
	void PushState(ofxGameState* state);

	// Pop the GameState at the top of the stack
	void PopState();

	// These wrap around ofApp.h functions.
	void setup();
	void update();
	void draw();

	inline static ofxGameEngine* Instance() {
		return &instance;
	}
};
#endif /* OFX_GAME_ENGINE_H */